// TMDriver: bridge between TMInterface 2.1 and the TMDriverAI Python server.
//
// The plugin listens on 127.0.0.1:8478; `python tmdriver.py serve` connects to it.
// Wire format (little endian) is mirrored in src/tmdriver/protocol.py - change both together.
//
// ACTION bits: 1 up, 2 down, 4 left, 8 right, 16 analog steer valid, 32 analog gas valid,
// 64 respawn (queued for the next tick via SimulationManager::Respawn).
//
// Modes (chosen with the window buttons, or set by Python):
//   IDLE    nothing is sent
//   RECORD  a STEP is streamed every physics tick while you drive (no waiting)
//   DRIVE   a STEP is sent every tick and the plugin waits for Python's ACTION
//   TEST    like DRIVE; Python runs the self test (simulation-only speed, determinism)

const int PROTOCOL = 11;
const string HOST = "127.0.0.1";
const uint16 PORT = 8478;      // the main instance; helpers take the next free ports
const int MAX_INSTANCES = 8;   // (Python: instances.MAX_INSTANCES)
const uint STEP_TIMEOUT_MS = 5000;

// plugin -> python
const int P_HELLO = 1;
const int P_MAP = 2;
const int P_STEP = 3;
const int P_UI = 4;
const int P_BENCH = 5;
const int P_PING = 6;     // heartbeat (idle only): a failed write reveals a dead client
const int P_TREC = 7;     // batch playback: one tick's state (STEP layout), no answer expected
const int P_JOB = 9;      // a tool button: int job (1 drive, 2 train, 3 re-simulate, 4 show best run),
                          // int TMX id (0 = the current map), int rounds, int minutes, int flags
                          // (1 GPU, 2 per-tick re-simulation). The Python server opens the job in its own console.
const int JOB_DRIVE = 1;
const int JOB_TRAIN = 2;
const int JOB_RESIM = 3;
const int JOB_SHOW = 4;
const int JOB_LAUNCH = 5;      // start more game instances (rounds field = how many)
const int P_TEND = 8;     // batch playback finished: int reason (0 end of inputs, 1 finish, 2 frozen time), int race time

// python -> plugin
const int C_ACTION = 10;
const int C_SPEED = 11;
const int C_RESTART = 12;
const int C_STATUS = 13;
const int C_SIMONLY = 14;
const int C_SAVE = 15;
const int C_REWIND = 16;
const int C_MODE = 17;
const int C_BENCH = 18;
const int C_EXEC = 19;    // console command, run from Render() (e.g. "map <file>")
const int C_PLAY = 21;    // int n, n x (int steer, int gas, int bits) for race times 0, 10, ...:
                          // sent before an ACTION; from the next tick the plugin plays them itself,
                          // streams P_TREC, sends P_TEND, and gives the last tick to Python as a STEP
const int MAX_PLAY = 60000;
const int C_DRAW = 20;    // int n, n x (float x, y, z), float size: show a path as trigger boxes (n = 0 clears)
const int MAX_DRAW = 600;
const int C_WAIT = 22;    // (during a STEP) Python is busy, e.g. learning between rounds: extend the timeout.
                          // The game stays frozen in the step, as in a bruteforce run.

const int MODE_IDLE = 0;
const int MODE_RECORD = 1;
const int MODE_DRIVE = 2;
const int MODE_TEST = 3;

const int NUM_SLOTS = 8;
const int UNKNOWN_INPUT = 999999;

Net::Socket@ server = null;
Net::Socket@ client = null;

int mode = MODE_IDLE;
string status = "Start TMDriver_starten.bat in the TMDriverAI folder.";
float uiSpeed = 1.0f;

// tool settings
int jobMap = 0;
int jobRounds = 20;
float jobHours = 3.0f;
bool jobGpu = true;
bool jobPerTick = false;
bool jobLine = false;     // give the AI the reference line (fastest TMX replay)
bool jobShowBest = true;  // Train: show every new best run in the game

// Requests made outside a physics step are applied at the start of the next OnRunStep.
bool pendingRestart = false;
bool pendingRelease = false;
float pendingSpeed = -1.0f;
array<string> pendingExec;   // console commands, run in order from Render()
                             // (a single slot lost commands at low frame rates)
array<int> drawnIds;      // trigger ids this plugin created to show a path

// batch playback (re-simulation without a Python round trip per tick, like TMI's bruteforce)
bool playing = false;
bool playArmed = false;   // C_PLAY received inside a STEP: playback starts with the next tick
array<int> playSteer;
array<int> playGas;
array<int> playBits;
int playLastT = -1000000;

void ClearDrawn()
{
    for (uint i = 0; i < drawnIds.Length; i++) {
        RemoveTrigger(drawnIds[i]);
    }
    drawnIds.Resize(0);
}
int pendingSimOnly = -1;

string sentUid = "";
int listenPort = -1;
int jobHelpers = 2;
int lastRaceTime = -1000000;
bool finishSent = false;
array<SimulationState@> slots(8);   // NUM_SLOTS
int lastSteer = UNKNOWN_INPUT;
int lastGas = UNKNOWN_INPUT;

int benchLeft = 0;
int benchTicks = 0;
uint64 benchStart = 0;
uint simTicks = 0;
uint64 lastPing = 0;

PluginInfo@ GetPluginInfo()
{
    PluginInfo info;
    info.Author = "cheatoskar";
    info.Name = "TMDriver";
    info.Description = "TMDriverAI: lets a Python driver record, drive and test via a local socket";
    info.Version = "0.1";
    return info;
}

void Main()
{
    // the game's user folder (with Tracks\Challenges), '' = Python detects it.
    // Set it in the TMI console: set tmdriver_game_folder C:\Users\...\Documents\TrackMania
    RegisterVariable("tmdriver_game_folder", "");
    // Several game instances: each takes the first free port (the first one started is the
    // main instance on PORT, helpers get PORT + 1, + 2, ...; Python finds them by trying).
    for (int k = 0; k < MAX_INSTANCES; k++) {
        @server = Net::Socket();
        if (server.Listen(HOST, uint16(PORT + k))) {
            listenPort = PORT + k;
            break;
        }
        @server = null;
    }
    if (server !is null) {
        log("TMDriver: listening on " + HOST + ":" + listenPort
            + (listenPort == PORT ? " (main instance)" : " (helper instance)"));
    } else {
        log("TMDriver: no free port from " + int(PORT), Severity::Error);
    }
}

void OnDisabled()
{
    @client = null;
    @server = null;
}

string ModeName(int m)
{
    if (m == MODE_RECORD) return "recording";
    if (m == MODE_DRIVE) return "AI driving (live)";
    if (m == MODE_TEST) return "job running";
    return "ready";
}

// ---------------------------------------------------------------- connection

void Drop(const string&in why)
{
    log("TMDriver: " + why, Severity::Warning);
    @client = null;
    status = why;
    if (mode != MODE_IDLE) {
        mode = MODE_IDLE;
        pendingRelease = true;
    }
    pendingSimOnly = 0;
    pendingSpeed = 1.0f;
}

bool WaitBytes(uint n, uint64 deadline)
{
    while (client !is null && client.Available < n) {
        if (Time::Now > deadline) {
            return false;
        }
    }
    return client !is null;
}

void WriteStr(const string&in s)
{
    client.Write(int(s.Length));
    if (s.Length > 0) {
        client.Write(s);
    }
}

void SendJob(int job)
{
    if (client is null) return;
    int flags = (jobGpu ? 1 : 0) | (jobPerTick ? 2 : 0) | (jobLine ? 4 : 0) | (jobShowBest ? 8 : 0);
    client.Write(P_JOB);
    client.Write(job);
    client.Write(jobMap);
    client.Write(jobRounds);
    client.Write(int(jobHours * 60.0f + 0.5f));
    client.Write(flags);
    WriteStr(GetVariableString("tmdriver_game_folder"));
    status = "Job started: a console window opens ...";
}

void Tip(const string&in text)
{
    if (UI::IsItemHovered()) {
        UI::BeginTooltip();
        UI::Text(text);
        UI::EndTooltip();
    }
}

void SendUi(int m)
{
    if (client is null) return;
    client.Write(P_UI);
    client.Write(m);
    client.Write(uiSpeed);
}

// Reads and executes one message. `simManager` is null when called outside a physics
// step; commands that need the simulation are then deferred or ignored.
// Returns the message type, or -1 on timeout / lost connection.
int ReadMessage(SimulationManager@ simManager, uint64 deadline)
{
    if (!WaitBytes(4, deadline)) return -1;
    int type = client.ReadInt32();

    if (type == C_ACTION) {
        if (!WaitBytes(12, deadline)) return -1;
        int steer = client.ReadInt32();
        int gas = client.ReadInt32();
        int bits = client.ReadInt32();
        if (simManager !is null) {
            ApplyAction(simManager, steer, gas, bits);
        }
    } else if (type == C_SPEED) {
        if (!WaitBytes(4, deadline)) return -1;
        float s = client.ReadFloat();
        if (simManager !is null) {
            simManager.SetSpeed(s);
        } else {
            pendingSpeed = s;
        }
    } else if (type == C_RESTART) {
        if (simManager !is null) {
            simManager.GiveUp();
            finishSent = false;
            lastSteer = UNKNOWN_INPUT;
            lastGas = UNKNOWN_INPUT;
        } else {
            pendingRestart = true;
        }
    } else if (type == C_STATUS) {
        if (!WaitBytes(4, deadline)) return -1;
        int n = client.ReadInt32();
        if (n > 0) {
            if (!WaitBytes(uint(n), deadline)) return -1;
            status = client.ReadString(uint(n));
        } else {
            status = "";
        }
    } else if (type == C_SIMONLY) {
        if (!WaitBytes(4, deadline)) return -1;
        int on = client.ReadInt32();
        if (simManager !is null) {
            simManager.SimulationOnly = (on != 0);
        } else {
            pendingSimOnly = on;
        }
    } else if (type == C_SAVE) {
        if (!WaitBytes(4, deadline)) return -1;
        int slot = client.ReadInt32();
        if (simManager !is null && slot >= 0 && slot < NUM_SLOTS) {
            @slots[slot] = simManager.SaveState();
        } else {
            log("TMDriver: SAVE ignored (slot " + slot + ")", Severity::Warning);
        }
    } else if (type == C_REWIND) {
        if (!WaitBytes(4, deadline)) return -1;
        int slot = client.ReadInt32();
        if (simManager !is null && slot >= 0 && slot < NUM_SLOTS && slots[slot] !is null) {
            simManager.RewindToState(slots[slot]);
            finishSent = false;
            lastRaceTime = -1000000;
            lastSteer = UNKNOWN_INPUT;
            lastGas = UNKNOWN_INPUT;
        } else {
            log("TMDriver: REWIND ignored (slot " + slot + ")", Severity::Warning);
        }
    } else if (type == C_MODE) {
        if (!WaitBytes(4, deadline)) return -1;
        int m = client.ReadInt32();
        if (m == MODE_IDLE && mode != MODE_IDLE) {
            pendingRelease = true;
        }
        mode = m;
    } else if (type == C_EXEC) {
        if (!WaitBytes(4, deadline)) return -1;
        int n = client.ReadInt32();
        if (n > 0) {
            if (!WaitBytes(uint(n), deadline)) return -1;
            pendingExec.Add(client.ReadString(uint(n)));
        }
    } else if (type == C_PLAY) {
        if (!WaitBytes(4, deadline)) return -1;
        int n = client.ReadInt32();
        if (n < 1 || n > MAX_PLAY) {
            Drop("PLAY with " + n + " ticks: protocol mismatch");
            return -1;
        }
        playSteer.Resize(n);
        playGas.Resize(n);
        playBits.Resize(n);
        for (int i = 0; i < n; i++) {
            if (!WaitBytes(12, deadline + 5000)) return -1;
            playSteer[i] = client.ReadInt32();
            playGas[i] = client.ReadInt32();
            playBits[i] = client.ReadInt32();
        }
        playArmed = true;
        playLastT = -1000000;
    } else if (type == C_DRAW) {
        if (!WaitBytes(4, deadline)) return -1;
        int n = client.ReadInt32();
        if (n < 0 || n > MAX_DRAW) {
            Drop("DRAW with " + n + " points: protocol mismatch");
            return -1;
        }
        if (!WaitBytes(uint(n * 12 + 4), deadline)) return -1;
        ClearDrawn();
        array<vec3> pts;
        for (int i = 0; i < n; i++) {
            float x = client.ReadFloat();
            float y = client.ReadFloat();
            float z = client.ReadFloat();
            pts.Add(vec3(x, y, z));
        }
        float size = client.ReadFloat();
        for (int i = 0; i < n; i++) {
            vec3 corner = vec3(pts[i].x - size * 0.5f, pts[i].y - size * 0.5f, pts[i].z - size * 0.5f);
            drawnIds.Add(SetTrigger(Trigger3D(corner, vec3(size, size, size))));
        }
    } else if (type == C_BENCH) {
        if (!WaitBytes(4, deadline)) return -1;
        int n = client.ReadInt32();
        if (simManager !is null && n > 0) {
            benchLeft = n;
            benchTicks = n;
            benchStart = Time::Now;
        }
    } else if (type == C_WAIT) {
        // no payload: the caller extends its deadline
    } else {
        Drop("Unknown message " + type + ": protocol mismatch");
        return -1;
    }
    return type;
}

// ---------------------------------------------------------------- inputs

void ApplyAction(SimulationManager@ simManager, int steer, int gas, int bits)
{
    simManager.SetInputState(InputType::Up, (bits & 1) != 0 ? 1 : 0);
    simManager.SetInputState(InputType::Down, (bits & 2) != 0 ? 1 : 0);
    simManager.SetInputState(InputType::Left, (bits & 4) != 0 ? 1 : 0);
    simManager.SetInputState(InputType::Right, (bits & 8) != 0 ? 1 : 0);

    // Analog inputs are only written when requested or when they must be reset to 0,
    // so keyboard-style actions do not fight a stale analog value.
    int s = (bits & 16) != 0 ? steer : 0;
    if ((bits & 16) != 0 || s != lastSteer) {
        simManager.SetInputState(InputType::Steer, s);
        lastSteer = s;
    }
    int g = (bits & 32) != 0 ? gas : 0;
    if ((bits & 32) != 0 || g != lastGas) {
        simManager.SetInputState(InputType::Gas, g);
        lastGas = g;
    }
    if ((bits & 64) != 0) {
        simManager.Respawn();
    }
}

void ReleaseInputs(SimulationManager@ simManager)
{
    ApplyAction(simManager, 0, 0, 0);
}

// ---------------------------------------------------------------- state

void WriteVec(const vec3&in v)
{
    client.Write(v.x);
    client.Write(v.y);
    client.Write(v.z);
}

void WriteWheel(TM::SceneVehicleCar::SimulationWheel@ w)
{
    if (w is null) {
        client.Write(0.0f);
        client.Write(0);
        return;
    }
    int b = 0;
    if (w.RTState.HasGroundContact) b |= 1;
    if (w.RTState.IsSliding) b |= 2;
    b |= (int(w.RTState.get_ContactMaterialId()) & 255) << 8;
    client.Write(w.RTState.DamperAbsorbVal);
    client.Write(b);
}

bool SendStep(SimulationManager@ simManager, int msgType = P_STEP)
{
    int flags = 0;
    if (simManager.PlayerInfo.RaceFinished) flags |= 1;
    if (simManager.SimulationOnly) flags |= 2;
    if (simManager.TickTime > simManager.RaceTime) flags |= 4;

    if (!client.Write(msgType)) return false;
    client.Write(simManager.RaceTime);
    client.Write(flags);
    client.Write(int(simManager.PlayerInfo.CurCheckpointCount));

    WriteVec(simManager.Dyna.CurrentState.Location.Position);
    mat3 rot = simManager.Dyna.CurrentState.Location.Rotation;
    WriteVec(rot.x);
    WriteVec(rot.y);
    WriteVec(rot.z);
    WriteVec(simManager.Dyna.CurrentState.LinearSpeed);
    WriteVec(simManager.Dyna.CurrentState.AngularSpeed);

    SimulationWheels@ wheels = simManager.Wheels;
    WriteWheel(wheels.FrontLeft);
    WriteWheel(wheels.FrontRight);
    WriteWheel(wheels.BackRight);
    WriteWheel(wheels.BackLeft);

    TM::SceneVehicleCar@ car = simManager.SceneVehicleCar;
    if (car !is null) {
        client.Write(car.CarEngine.Gear);
        client.Write(car.CarEngine.ActualRPM);
    } else {
        client.Write(0);
        client.Write(0.0f);
    }
    client.Write(int(simManager.PlayerInfo.DisplaySpeed));

    InputState inp = simManager.GetInputState();
    int ib = 0;
    if (inp.Up) ib |= 1;
    if (inp.Down) ib |= 2;
    if (inp.Left) ib |= 4;
    if (inp.Right) ib |= 8;
    client.Write(inp.Steer);
    client.Write(inp.Gas);
    return client.Write(ib);
}

void SendMapIfNew()
{
    TM::GameCtnChallenge@ ch = GetCurrentChallenge();
    if (ch is null) return;
    string uid = ch.Uid;
    if (uid == sentUid) return;
    sentUid = uid;

    const array<TM::GameCtnBlock@>@ blocks = ch.Blocks;
    int count = (blocks !is null) ? int(blocks.Length) : 0;
    client.Write(P_MAP);
    WriteStr(uid);
    WriteStr(ch.Name);
    WriteStr(ch.Author);
    client.Write(count);
    for (int i = 0; i < count; i++) {
        const TM::GameCtnBlock@ b = blocks[i];
        if (b is null) {
            // keep the announced count: write an empty placeholder block
            WriteStr("");
            client.Write(0);
            client.Write(0);
            client.Write(0);
            client.Write(0);
            client.Write(3);
            continue;
        }
        WriteStr(b.Name);
        client.Write(int(b.Coord.x));
        client.Write(int(b.Coord.y));
        client.Write(int(b.Coord.z));
        client.Write(int(b.Dir));
        client.Write(int(b.WayPointType));
    }
}

// ---------------------------------------------------------------- callbacks

void ApplyPending(SimulationManager@ simManager)
{
    if (pendingSimOnly >= 0) {
        simManager.SimulationOnly = (pendingSimOnly != 0);
        pendingSimOnly = -1;
    }
    if (pendingSpeed > 0.0f) {
        simManager.SetSpeed(pendingSpeed);
        pendingSpeed = -1.0f;
    }
    if (pendingRelease) {
        ReleaseInputs(simManager);
        pendingRelease = false;
    }
}

void RunBench(SimulationManager@ simManager)
{
    simManager.SetInputState(InputType::Up, 1);
    benchLeft--;
    if (benchLeft == 0) {
        uint64 elapsed = Time::Now - benchStart;
        client.Write(P_BENCH);
        client.Write(benchTicks);
        client.Write(int(elapsed));
        client.Write(simManager.RaceTime);
    }
}

void OnRunStep(SimulationManager@ simManager)
{
    ApplyPending(simManager);

    if (simManager.SimulationOnly) {
        simTicks++;
        // Render() is not called in simulation-only mode; keep the game responsive.
        if (simTicks % 1000 == 0) {
            Graphics::ForceGameRender();
        }
    }

    if (client is null) return;

    if (pendingRestart) {
        pendingRestart = false;
        simManager.GiveUp();
        finishSent = false;
        lastSteer = UNKNOWN_INPUT;
        lastGas = UNKNOWN_INPUT;
        return;
    }

    int t = simManager.RaceTime;
    if (t < lastRaceTime) {
        finishSent = false;  // a new attempt started (restart or a newly loaded map):
        sentUid = "";        // announce the map again, so Python can tell a reload happened
    }
    lastRaceTime = t;

    SendMapIfNew();

    if (benchLeft > 0) {
        RunBench(simManager);
        return;
    }

    if (playing && !RunPlayback(simManager, t)) return;

    // Commands between steps (mode, status, drawing) normally arrive in Render(), which the
    // game does not call while its window is minimized: read them here as well.
    while (client !is null && client.Available >= 4) {
        if (ReadMessage(null, Time::Now + 500) == -1) {
            if (client !is null) Drop("Incomplete message from Python");
            break;
        }
    }
    if (client is null) return;

    if (mode == MODE_IDLE || t < -10 || finishSent) return;

    bool finished = simManager.PlayerInfo.RaceFinished;
    if (!SendStep(simManager)) {
        Drop("Lost the connection to Python");
        return;
    }
    if (finished) finishSent = true;
    if (mode == MODE_RECORD) return;

    // DRIVE / TEST: execute Python's commands until its ACTION for this tick arrives.
    uint64 deadline = Time::Now + STEP_TIMEOUT_MS;
    while (client !is null) {
        int type = ReadMessage(simManager, deadline);
        if (type == C_ACTION) break;
        if (type == C_WAIT) deadline = Time::Now + STEP_TIMEOUT_MS;
        if (type == -1) {
            if (client !is null) Drop("Python did not answer (timeout)");
            simManager.SimulationOnly = false;
            ReleaseInputs(simManager);
            break;
        }
    }
    if (playArmed) {            // the ACTION above is the input for this tick; the rest is ours
        playArmed = false;
        playing = true;
        playLastT = -1000000;   // a REWIND may have come with it: t is no longer the race time
    }
}

// One tick of batch playback: report the state, then apply the recorded input for this
// race time (the same order as STEP -> ACTION). Returns true when playback ends (finish,
// frozen race time, or no input left): P_TEND goes out and this tick continues as an
// ordinary STEP, so Python decides what happens next (REWIND + the next PLAY, or stop).
// The plugin never rewinds by itself: a rewind here would be followed by one simulated
// tick without input, and the next run would not start from the saved state.
bool RunPlayback(SimulationManager@ simManager, int t)
{
    int reason = -1;
    if (t == playLastT) reason = 2;
    else if (simManager.PlayerInfo.RaceFinished) reason = 1;
    else if (t < 0 || t / 10 >= int(playSteer.Length)) reason = 0;
    playLastT = t;
    if (reason >= 0) {
        playing = false;
        client.Write(P_TEND);
        client.Write(reason);
        client.Write(t);
        return true;
    }
    if (!SendStep(simManager, P_TREC)) {
        Drop("Lost the connection to Python");
        playing = false;
        return false;
    }
    int i = t / 10;
    ApplyAction(simManager, playSteer[i], playGas[i], playBits[i]);
    return false;
}

void StartMode(int m)
{
    if (m != MODE_IDLE) {
        pendingRestart = true;
    } else if (mode != MODE_IDLE) {
        pendingRelease = true;
        pendingSimOnly = 0;
        pendingSpeed = 1.0f;
    }
    mode = m;
    SendUi(m);
}

void Render()
{
    // A new connection always replaces the current one: a Python process that died while
    // no race was running is otherwise never noticed, and would block every later client.
    if (server !is null) {
        Net::Socket@ s = server.Accept(0);
        if (s !is null) {
            if (client !is null) {
                log("TMDriver: new Python connection replaces the old one", Severity::Warning);
                if (mode != MODE_IDLE) {
                    mode = MODE_IDLE;
                    pendingRelease = true;
                    pendingSimOnly = 0;
                    pendingSpeed = 1.0f;
                }
            }
            @client = s;
            lastPing = Time::Now;
            client.NoDelay = true;
            sentUid = "";
            status = "Python connected";
            client.Write(P_HELLO);
            client.Write(PROTOCOL);
            log("TMDriver: Python connected (" + client.RemoteIP + ")", Severity::Success);
        }
    }

    // Heartbeat: detects a Python process that went away while nothing else is sent.
    if (client !is null && Time::Now - lastPing > 1000) {
        lastPing = Time::Now;
        if (!client.Write(P_PING)) {
            Drop("Lost the connection to Python");
        }
    }

    // Status text and mode changes can arrive between races.
    while (client !is null && client.Available >= 4) {
        if (ReadMessage(null, Time::Now + 500) == -1) {
            if (client !is null) Drop("Incomplete message from Python");
            break;
        }
    }

    // Console commands (map loads) run here, never inside a physics step.
    for (uint i = 0; i < pendingExec.Length; i++) {
        log("TMDriver: " + pendingExec[i]);
        ExecuteCommand(pendingExec[i]);
    }
    pendingExec.Resize(0);

    if (UI::Begin("DeltaZero")) {
        UI::Text(client is null ? "Python: not connected" : "Python: connected (" + ModeName(mode) + ")");
        if (listenPort != int(PORT)) {
            UI::TextWrapped("Helper instance (port " + listenPort + "): the main window starts the jobs.");
        }
        UI::TextWrapped(status);
        if (client is null) {
            UI::TextWrapped("Start 'TMDriver_starten.bat' in the TMDriverAI folder first.");
        }
        UI::Separator();
        jobMap = UI::InputInt("Map (TMX id)", jobMap, 0);
        if (jobMap < 0) jobMap = 0;
        Tip("0 = the map that is loaded now. Otherwise the TMX id, e.g. 10036840 for lolsport.");
        string gameFolder = GetVariableString("tmdriver_game_folder");
        UI::TextWrapped("Game folder: " + (gameFolder == "" ? "auto" : gameFolder));
        Tip("Where the game keeps Tracks\\Challenges (maps are downloaded there). auto = Python picks "
            "Documents\\TmForever or Documents\\TrackMania, whichever was used last. To set it, type in the "
            "TMI console: set tmdriver_game_folder C:\\Users\\<you>\\Documents\\TrackMania");
        jobGpu = UI::Checkbox("Use GPU (CUDA)", jobGpu);
        Tip("Off = CPU. The laptop has no CUDA GPU.");
        jobLine = UI::Checkbox("Use reference line (TMX replay)", jobLine);
        Tip("On: the AI sees the line of the fastest TMX replay of this map (downloaded if needed) "
            "and follows it. Off: it drives from the blocks alone; progress then counts the track "
            "blocks reached and the checkpoints.");
        UI::BeginDisabled(client is null || mode != MODE_IDLE);
        if (UI::Button("Drive")) SendJob(JOB_DRIVE);
        Tip("The AI drives the map once without rendering (the game freezes briefly), draws its "
            "line as small boxes, then drives it visibly.");
        UI::SameLine();
        if (UI::Button("Show best run")) SendJob(JOB_SHOW);
        Tip("Plays the best run that 'Train' found on this map.");
        jobRounds = UI::SliderInt("Rounds", jobRounds, 1, 200);
        Tip("Training rounds. Each round: 7 runs (1 normal + 6 with random variations), then the AI "
            "learns from the best ones. The game stays frozen while it trains.");
        jobShowBest = UI::Checkbox("Show new best runs", jobShowBest);
        Tip("Train: after a round with a new best run, drive it visibly in the game. Respawn while it "
            "is shown skips it and the next round starts at once.");
        if (UI::Button("Train")) SendJob(JOB_TRAIN);
        Tip("The AI practises this map and gets faster. Every new best run is shown in the game, "
            "then training continues.");
        UI::EndDisabled();
        UI::SameLine();
        UI::BeginDisabled(client is null);
        if (UI::Button("Stop")) StartMode(MODE_IDLE);
        Tip("Stops the running job or mode.");
        UI::EndDisabled();
        UI::TextWrapped("Each button opens a console with live progress.");

        if (UI::CollapsingHeader("Helper instances")) {
            jobHelpers = UI::SliderInt("Helpers", jobHelpers, 1, MAX_INSTANCES - 1);
            UI::BeginDisabled(client is null || mode != MODE_IDLE);
            if (UI::Button("Launch helpers")) {
                int saved = jobRounds;
                jobRounds = jobHelpers;
                SendJob(JOB_LAUNCH);
                jobRounds = saved;
            }
            Tip("Starts more game instances (TMLoader profile DeltaZero). Log in to each; the status "
                "line here shows how many are ready. Re-simulate then uses all of them.");
            UI::EndDisabled();
        }
        if (UI::CollapsingHeader("Data: re-simulate replays")) {
            jobHours = UI::SliderFloat("Hours", jobHours, 0.1f, 24.0f, "%.1f h");
            jobPerTick = UI::Checkbox("Per-tick mode (slower, verified)", jobPerTick);
            Tip("Default: the plugin plays each replay's inputs itself (batch mode). Per tick: Python "
                "answers every tick; use it if batch mode reports runs that are not exact.");
            UI::BeginDisabled(client is null || mode != MODE_IDLE);
            if (UI::Button("Re-simulate replays")) SendJob(JOB_RESIM);
            Tip("Plays TMX replays in the game and saves the full physics state (data/resim). "
                "Map (TMX id) > 0: that map and its 5 fastest replays (downloaded if needed). 0: all "
                "downloaded maps if the bulk list exists, else the open map. Stops after the hours above.");
            UI::EndDisabled();
        }
        if (UI::CollapsingHeader("Advanced")) {
            UI::BeginDisabled(client is null);
            if (UI::Button("Record (drive yourself)")) StartMode(MODE_RECORD);
            Tip("Records your own run as a reference line for the AI.");
            UI::SameLine();
            if (UI::Button("Self-test")) StartMode(MODE_TEST);
            Tip("Checks the game conventions (input signs, replay alignment). Run once per install.");
            if (UI::Button("AI drive (live)")) StartMode(MODE_DRIVE);
            Tip("The AI drives in real time, without planning or drawing a line.");
            uiSpeed = UI::SliderFloat("Speed", uiSpeed, 1.0f, 10.0f, "%.1fx");
            Tip("Game speed for 'AI drive (live)'.");
            UI::EndDisabled();
        }
    }
    UI::End();
}

/* A small C API over TMNF-C for DeltaZero's virtual game (ctypes, one DLL):
 * n cars on one track, stepped together one 10 ms tick at a time with any inputs,
 * each car's state readable, capturable and restorable (save / rewind).
 *
 * Every world is created from the vehicle snapshot, respawned at the map's start and stepped
 * once with no input, which gives the map's race-time-10 state exactly (tmnfc_run.c). The route
 * file only carries the start (compat/route.py writes a minimal one); checkpoints and the
 * finish are DeltaZero's business.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#define API __declspec(dllexport)
#else
#define API
#endif

#include "physics.h"
#include "race.h"
#include "route.h"
#include "track.h"
#include "vec_env.h"
#include "vehicle.h"
#include "world.h"

typedef struct {
	TmnfTrack *track;
	TmnfRoute *route;
	TmnfWorld **worlds;
	TmnfPhysicsWorld **physics;
	uint32_t *players;
	uint32_t n;
	TmnfVecEnv env;
	TmnfObservation *obs;
	TMNFRaceInputs *inputs;
} TmcSim;

/* Per car: pos[3] rot[9] vel[3] ang_vel[3] damper[4] contact[4] sliding[4] material[4] rpm gear */
enum { TMC_STATE_FLOATS = 3 + 9 + 3 + 3 + 4 + 4 + 4 + 4 + 2 };

static int parse_sha(const char *hex, uint8_t out[32])
{
	if (hex == NULL || strlen(hex) != 64)
		return -1;
	for (int i = 0; i < 32; ++i) {
		unsigned v;
		if (sscanf(hex + 2 * i, "%2x", &v) != 1)
			return -1;
		out[i] = (uint8_t)v;
	}
	return 0;
}

extern int tmnf_skip_spawn_check;

API uint32_t tmc_state_floats(void) { return TMC_STATE_FLOATS; }
API uint32_t tmc_snapshot_size(void) { return (uint32_t)sizeof(TmnfEnvSnapshot); }
API uint32_t tmc_input_size(void) { return (uint32_t)sizeof(TMNFRaceInputs); }

API TmcSim *tmc_open(const char *track_path, const char *vehicle_path, const char *route_path,
	const char *sha_hex, const float *spawn12, uint32_t n, uint32_t threads)
{
#ifdef _WIN32
	SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
	uint8_t sha[32];
	if (n == 0 || parse_sha(sha_hex, sha) != 0)
		return NULL;
	TmcSim *s = calloc(1, sizeof(*s));
	s->n = n;
	s->track = TmnfTrack_Load(track_path, sha);
	s->route = TmnfRoute_Load(route_path, sha);
	s->worlds = calloc(n, sizeof(*s->worlds));
	s->physics = calloc(n, sizeof(*s->physics));
	s->players = calloc(n, sizeof(*s->players));
	s->obs = calloc(n, sizeof(*s->obs));
	s->inputs = calloc(n, sizeof(*s->inputs));
	for (uint32_t i = 0; i < n; ++i) {
		TmnfWorld *w = World_Create(s->track, vehicle_path);
		if (spawn12 != NULL) {
			TMNFRaceInputs none;
			memset(&none, 0, sizeof(none));
			World_Respawn(w, (const GmIso4 *)spawn12);
			CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&none, World_GetPlayerVehicle(w));
			World_AdvanceTimer(w, 10);
			CHmsZoneDynamic_PhysicsStep2(World_GetPhysicsWorld(w), 10);
		}
		s->worlds[i] = w;
		s->physics[i] = World_GetPhysicsWorld(w);
	}
	TmnfVecEnvConfig config = TmnfVecEnv_DefaultConfig();
	config.max_race_ticks = 360000;          /* one hour: DeltaZero ends episodes itself */
	config.horizon_ticks = 360000;
	config.off_track_grace_ticks = 360000;
	config.stuck_grace_ticks = 360000;
	config.thread_count = threads ? threads : 1;
	config.autoreset_mode = TMNF_AUTORESET_SAME_STEP;
	config.action_space = TMNF_ACTION_SPACE_ANALOG;
	config.respawn_action = 1;
	tmnf_skip_spawn_check = spawn12 != NULL;
	TmnfVecEnv_Init(&s->env, s->physics, s->players, n, s->route, &config);
	return s;
}

API void tmc_close(TmcSim *s)
{
	if (s == NULL)
		return;
	TmnfVecEnv_Destroy(&s->env);
	for (uint32_t i = 0; i < s->n; ++i)
		World_Destroy(s->worlds[i]);
	TmnfRoute_Unload(s->route);
	TmnfTrack_Unload(s->track);
	free(s->worlds);
	free(s->physics);
	free(s->players);
	free(s->obs);
	free(s->inputs);
	free(s);
}

/* One tick for every car: inputs is n TMNFRaceInputs. A respawn press (input.respawn) puts the
 * car at the last respawnable checkpoint first (the race restarts when there is none: the
 * caller sees race state restarted = 1 and resets that car). Then physics, then the race
 * bookkeeping: checkpoints in any order, the finish once all are taken. */
API void tmc_step(TmcSim *s, const TMNFRaceInputs *inputs, uint8_t *restarted)
{
	/* The environment's tick applies a respawn press itself (to the last respawnable
	 * checkpoint); with none passed yet the game restarts the race instead, which the
	 * environment treats as an error: that press is dropped here and reported. */
	TMNFRaceInputs *in = s->inputs;
	memcpy(in, inputs, s->n * sizeof(*in));
	for (uint32_t i = 0; i < s->n; ++i) {
		restarted[i] = 0;
		if (in[i].respawn != 0 && TmnfRace_RespawnLocation(&s->env.race_states[i]) == NULL) {
			in[i].respawn = 0;
			restarted[i] = 1;
		}
	}
	TmnfVecEnv_Step(&s->env, in, 10, s->obs);
	for (uint32_t i = 0; i < s->n; ++i) {
		TmnfRaceState *race = &s->env.race_states[i];
		if (race->finished)
			continue;
		const TmnfPhysicsWorld *physics = s->physics[i];
		(void)TmnfRace_Step(s->route, race, physics->trigger_contacts,
			physics->corpora[s->players[i]].collision_corpus->live_iso);
	}
}

/* Race state of car i: checkpoints taken, finished, finish time (ms), respawn available. */
API void tmc_race(TmcSim *s, uint32_t i, uint32_t *out)
{
	const TmnfRaceState *r = &s->env.race_states[i];
	out[0] = r->visited_count;
	out[1] = r->finished;
	out[2] = r->finish_time_ms;
	out[3] = r->respawn_available;
}

/* Back to the start state (race time 10) for the cars with mask[i] != 0. */
API void tmc_reset(TmcSim *s, const uint8_t *mask)
{
	TmnfVecEnv_Reset(&s->env, mask, s->obs);
}

API void tmc_state(TmcSim *s, uint32_t i, float *out)
{
	const CHmsStateDyna *d = World_GetPlayerState(s->worlds[i]);
	const TmnfObservation *o = &s->obs[i];
	float *p = out;
	memcpy(p, &d->pos, 12); p += 3;
	memcpy(p, &d->rot, 36); p += 9;
	memcpy(p, &d->linVel, 12); p += 3;
	memcpy(p, &d->angVel, 12); p += 3;
	memcpy(p, o->wheel_damper, 16); p += 4;
	memcpy(p, o->wheel_contact, 16); p += 4;
	memcpy(p, o->wheel_sliding, 16); p += 4;
	memcpy(p, o->wheel_material, 16); p += 4;
	*p++ = o->engine_rpm;
	*p++ = (float)o->gear;
}

API void tmc_capture(TmcSim *s, uint32_t i, void *snapshot)
{
	TmnfVecEnv_CaptureIndices(&s->env, &i, 1, (TmnfEnvSnapshot *)snapshot);
}

API void tmc_restore(TmcSim *s, uint32_t i, const void *snapshot)
{
	TmnfVecEnv_RestoreIndices(&s->env, &i, 1, (const TmnfEnvSnapshot *)snapshot, &s->obs[i]);
}

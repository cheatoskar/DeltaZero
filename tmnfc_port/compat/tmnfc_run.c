/* Run an input schedule on a track, optionally moving the car to another start first.
 *
 *   tmnfc_run TRACK VEHICLE TRACK_SHA256 INPUTS OUTPUT [SPAWN12 [MODE]]
 *
 * INPUTS: TMNFRaceInputs records (tests/replay_tick.c input_file format).
 * OUTPUT: per tick int32 race time + float32 pos[3] (the state before the tick's step, like
 *         replay_tick --native-capture).
 * SPAWN12: "m0,...,m8,x,y,z" (GmIso4, row-major rotation then translation). The vehicle
 *          snapshot's state is its capture track's state at race time 10; with a spawn the car
 *          is respawned there (World_Respawn) and stepped once with no input, which should
 *          give the new track's race-time-10 state.
 * MODE: 0 = the extra step also advances the race timer, 1 = it does not.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "physics.h"
#include "race.h"
#include "track.h"
#include "vehicle.h"
#include "world.h"

static void fail(const char *m)
{
	fprintf(stderr, "tmnfc_run: %s\n", m);
	exit(1);
}

static void parse_sha(const char *hex, uint8_t out[32])
{
	if (strlen(hex) != 64)
		fail("sha256 must be 64 hex characters");
	for (int i = 0; i < 32; ++i) {
		unsigned v;
		if (sscanf(hex + 2 * i, "%2x", &v) != 1)
			fail("bad sha256");
		out[i] = (uint8_t)v;
	}
}

int main(int argc, char **argv)
{
	if (argc < 6)
		fail("usage: TRACK VEHICLE TRACK_SHA256 INPUTS OUTPUT [SPAWN12 [MODE]]");
	uint8_t sha[32];
	parse_sha(argv[3], sha);
	FILE *f = fopen(argv[4], "rb");
	if (!f)
		fail("cannot open inputs");
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	rewind(f);
	if (size <= 0 || size % sizeof(TMNFRaceInputs) != 0)
		fail("inputs are not whole records");
	uint32_t n = (uint32_t)(size / sizeof(TMNFRaceInputs));
	TMNFRaceInputs *in = malloc(size);
	if (fread(in, sizeof(*in), n, f) != n)
		fail("short read");
	fclose(f);

	TmnfTrack *track = TmnfTrack_Load(argv[1], sha);
	TmnfWorld *world = World_Create(track, argv[2]);
	if (argc >= 7) {
		GmIso4 spawn;
		float *v = (float *)&spawn;
		const char *p = argv[6];
		for (int i = 0; i < 12; ++i) {
			char *end;
			v[i] = strtof(p, &end);
			if (end == p)
				fail("SPAWN12 needs 12 comma-separated floats");
			p = *end == ',' ? end + 1 : end;
		}
		int mode = argc >= 8 ? atoi(argv[7]) : 0;
		World_Respawn(world, &spawn);
		TMNFRaceInputs none;
		memset(&none, 0, sizeof(none));
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&none, World_GetPlayerVehicle(world));
		if (mode == 0)
			World_AdvanceTimer(world, 10);
		CHmsZoneDynamic_PhysicsStep2(World_GetPhysicsWorld(world), 10);
	}
	FILE *out = fopen(argv[5], "wb");
	if (!out)
		fail("cannot open output");
	for (uint32_t k = 0; k < n; ++k) {
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&in[k], World_GetPlayerVehicle(world));
		const CHmsStateDyna *s = World_GetPlayerState(world);
		int32_t t = (int32_t)((k + 1) * 10);
		fwrite(&t, 4, 1, out);
		fwrite(&s->pos, 12, 1, out);
		World_AdvanceTimer(world, 10);
		CHmsZoneDynamic_PhysicsStep2(World_GetPhysicsWorld(world), 10);
	}
	fclose(out);
	World_Destroy(world);
	TmnfTrack_Unload(track);
	free(in);
	return 0;
}

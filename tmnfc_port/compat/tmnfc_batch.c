/* Many input schedules on one track, loading the track once.
 *
 *   tmnfc_batch TRACK VEHICLE TRACK_SHA256 SPAWN12|-
 *
 * Then one job per stdin line: "INPUTS OUTPUT". For each job a fresh world is created from the
 * vehicle snapshot; with SPAWN12 ("m0,...,m8,x,y,z") the car is respawned there and stepped
 * once with no input (the new track's race-time-10 state, see tmnfc_run.c). Answers one line
 * per job on stdout: "ok TICKS" or "error MESSAGE".
 *
 * OUTPUT, per tick (state before the tick's step, race time (k+1)*10):
 *   int32 race_time, float32 pos[3], rot[9], lin_vel[3], ang_vel[3]   (76 bytes)
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "physics.h"
#include "race.h"
#include "track.h"
#include "vehicle.h"
#include "world.h"

static void fail(const char *m)
{
	fprintf(stderr, "tmnfc_batch: %s\n", m);
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

static double now(void)
{
	struct timespec ts;
	timespec_get(&ts, TIME_UTC);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int run_job(const TmnfTrack *track, const char *vehicle, const GmIso4 *spawn,
	const char *in_path, const char *out_path, char *err, size_t err_size)
{
	FILE *f = fopen(in_path, "rb");
	if (!f) {
		snprintf(err, err_size, "cannot open %s", in_path);
		return -1;
	}
	fseek(f, 0, SEEK_END);
	long size = ftell(f);
	rewind(f);
	if (size <= 0 || size % sizeof(TMNFRaceInputs) != 0) {
		fclose(f);
		snprintf(err, err_size, "inputs are not whole records");
		return -1;
	}
	uint32_t n = (uint32_t)(size / sizeof(TMNFRaceInputs));
	TMNFRaceInputs *in = malloc(size);
	size_t got = fread(in, sizeof(*in), n, f);
	fclose(f);
	if (got != n) {
		free(in);
		snprintf(err, err_size, "short read");
		return -1;
	}
	TmnfWorld *world = World_Create(track, vehicle);
	if (spawn != NULL) {
		TMNFRaceInputs none;
		memset(&none, 0, sizeof(none));
		World_Respawn(world, spawn);
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&none, World_GetPlayerVehicle(world));
		World_AdvanceTimer(world, 10);
		CHmsZoneDynamic_PhysicsStep2(World_GetPhysicsWorld(world), 10);
	}
	FILE *out = fopen(out_path, "wb");
	if (!out) {
		World_Destroy(world);
		free(in);
		snprintf(err, err_size, "cannot open %s", out_path);
		return -1;
	}
	for (uint32_t k = 0; k < n; ++k) {
		CTrackManiaControlPlayerInput_UpdateVehicleStateFromInputs(&in[k], World_GetPlayerVehicle(world));
		const CHmsStateDyna *s = World_GetPlayerState(world);
		int32_t t = (int32_t)((k + 1) * 10);
		fwrite(&t, 4, 1, out);
		fwrite(&s->pos, 12, 1, out);
		fwrite(&s->rot, 36, 1, out);
		fwrite(&s->linVel, 12, 1, out);
		fwrite(&s->angVel, 12, 1, out);
		World_AdvanceTimer(world, 10);
		CHmsZoneDynamic_PhysicsStep2(World_GetPhysicsWorld(world), 10);
	}
	fclose(out);
	World_Destroy(world);
	free(in);
	return (int)n;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
	/* a crash must end the process, not open a "has stopped working" dialog that waits forever */
	SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
#endif
	if (argc != 5)
		fail("usage: TRACK VEHICLE TRACK_SHA256 SPAWN12|-");
	uint8_t sha[32];
	parse_sha(argv[3], sha);
	GmIso4 spawn, *spawn_ptr = NULL;
	if (strcmp(argv[4], "-") != 0) {
		float *v = (float *)&spawn;
		const char *p = argv[4];
		for (int i = 0; i < 12; ++i) {
			char *end;
			v[i] = strtof(p, &end);
			if (end == p)
				fail("SPAWN12 needs 12 comma-separated floats");
			p = *end == ',' ? end + 1 : end;
		}
		spawn_ptr = &spawn;
	}
	double t0 = now();
	TmnfTrack *track = TmnfTrack_Load(argv[1], sha);
	fprintf(stderr, "tmnfc_batch: track loaded in %.3f s\n", now() - t0);
	printf("ready\n");
	fflush(stdout);
	char line[4096], err[512];
	while (fgets(line, sizeof(line), stdin)) {
		line[strcspn(line, "\r\n")] = '\0';
		if (line[0] == '\0')
			break;
		char *sep = strchr(line, '\t');
		if (!sep) {
			printf("error expected INPUTS<TAB>OUTPUT\n");
			fflush(stdout);
			continue;
		}
		*sep = '\0';
		double j0 = now();
		int n = run_job(track, argv[2], spawn_ptr, line, sep + 1, err, sizeof(err));
		if (n < 0)
			printf("error %s\n", err);
		else
			printf("ok %d %.4f\n", n, now() - j0);
		fflush(stdout);
	}
	TmnfTrack_Unload(track);
	return 0;
}

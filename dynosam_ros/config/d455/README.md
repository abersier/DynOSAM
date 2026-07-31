# STALE — this directory is not currently loaded on the real robot

Confirmed 2026-07-29: `robot.launch.py`'s `params_path` launch arg defaults to `''`, which falls
through to `dynosam_ros`'s `get_default_dynosam_params_path()` — this resolves to the `dynosam`
package's own share directory, **not** this folder:

```
install/dynosam/share/dynosam/params/
```

Verified live against a running real-robot pipeline:

```bash
ros2 param get /dynosam/dynosam_node params_path
# String value is: /home/user/dev_ws/install/dynosam/share/dynosam/params/
```

Nothing in the documented real-robot launch command (`memory/commands.md` "Real robot
two-command startup" in `dyno_mpc`) passes `params_path:=` at all, so every file in this
directory — `FrontendParams.yaml`, `CameraParams.yaml`, `DatasetParams.yaml`, `ImuParams.yaml`,
`PipelineParams.yaml`, `*.flags` — has had **zero effect** on real-robot runs for as long as
that's been true. Edit `dynosam/params/` instead if you want a change to actually take effect,
unless you also pass `params_path:=<absolute path to this d455 dir>` explicitly at launch.

**Not yet reconciled**: this directory and `dynosam/params/` have diverged on several values
beyond the ones just discovered stale here (`ransac_iterations` 500 vs 300,
`min_features_per_frame` 150 vs 300, `min_dynamic_tracks` 20 vs 10, `max_object_depth` 50.0 vs
8.0, `max_background_depth` 80.0 vs 20) — any of those may represent real-robot tuning from past
sessions that was made here under the same mistaken assumption and never actually applied. Not
audited/fixed as of this note; worth a deliberate pass (or just deleting this directory and
standardizing on one config) once the current deadline pressure eases.

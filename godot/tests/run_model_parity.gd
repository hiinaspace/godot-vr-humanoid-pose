extends SceneTree
# Replays a recorded tracker stream through the running sidecar `learned` estimator
# (NDJSON live_capture.v1 over localhost TCP, the same StreamClient the live overlay uses) and
# compares the responses with the PyTorch outputs exported by
# `python -m validation.sprint.parity fixture`.
#
#   godot --headless --path godot --script res://tests/run_model_parity.gd -- \
#       --fixture=/abs/dir --port=8765 [--host=127.0.0.1] [--paced-seconds=6] \
#       [--max-p99-ms=30] [--report=/abs/report.json]
#
# Start the sidecar first (`validation/sprint/run_model_parity.py` does everything in one command):
#   python -m validation.scripts.run_live_capture_recorder --estimator learned \
#       --learned-model MODEL.onnx --learned-frame model --port 8765 --no-raw-log
#
# Phases: (A) 60 Hz stream replayed back-to-back, parity vs torch; (B) the 90 Hz resampled stream
# back-to-back, parity vs the runtime's own expected output; (R) reset reproducibility (explicit
# reset_estimator message, and an implicit reset by a backwards timestamp jump); (C) the 90 Hz stream
# paced in real time, stop-and-wait like live_overlay_debug.gd (one request in flight, a tick is
# skipped when the previous response has not arrived) to measure the achieved rate and round trip.
# Exit code 1 on any failed assertion.

const StreamClient = preload("res://addons/godot_vr_humanoid_pose/scripts/live_capture_stream_client.gd")
const RAW_DIM := 24

var _failures: Array[String] = []
var _args := {}
var _client
var _session := "parity-%d" % Time.get_ticks_usec()
var _request_id := 0
var _meta := {}
var _joints: Array = []
var _nj := 0
var _calibration := {}
var _stream60 := PackedFloat32Array()
var _exp_pos60 := PackedFloat32Array()
var _exp_quat60 := PackedFloat32Array()
var _stream90 := PackedFloat32Array()
var _t90 := PackedFloat64Array()
var _exp_pos90 := PackedFloat32Array()
var _exp_quat90 := PackedFloat32Array()
var _report := {}
var _tick_us := 11111


func _init() -> void:
	_parse_args()
	if not _load_fixture():
		_finish()
		return
	_client = StreamClient.new()
	_client.configure(str(_args.get("host", "127.0.0.1")), int(_args.get("port", "8765")))
	if not _connect():
		_failures.append("could not connect to the sidecar at %s:%s" % [_args.get("host", "127.0.0.1"), _args.get("port", "8765")])
		_finish()
		return
	_client.send_message(_client.build_session_message(_session, "run_model_parity"))

	_tick_us = int(round(1000000.0 / float(_args.get("paced-hz", "90"))))
	var latency_only := str(_args.get("latency-only", "0")) == "1"
	if latency_only:  # protocol-floor measurement against any estimator (no joint comparison)
		var lp := _paced(float(_args.get("paced-seconds", "6")))
		_report["C_paced"] = lp
		_print_paced(lp)
		_write_report()
		_finish()
		return
	var n60 := int(_meta["frames60"])
	var n90 := int(_meta["frames90"])
	var tol_pos := float(_meta["tol_pos_m"])
	var tol_rot := float(_meta["tol_rot_deg"])

	var a := _replay("A 60Hz", _stream60, null, n60, 60.0, _exp_pos60, _exp_quat60, tol_pos, tol_rot)
	var b := _replay("B 90Hz", _stream90, _t90, n90, 90.0, _exp_pos90, _exp_quat90, tol_pos, tol_rot)
	_report["A_60hz"] = a
	_report["B_90hz"] = b

	_test_reset_reproducibility(a)

	var paced := _paced(float(_args.get("paced-seconds", "6")))
	_report["C_paced"] = paced
	var max_p99 := float(_args.get("max-p99-ms", "30"))
	_assert_true("paced p99 round trip %.2f ms <= %.1f ms" % [paced["rtt_p99_ms"], max_p99], paced["rtt_p99_ms"] <= max_p99)
	_assert_true("paced run answered %d requests" % int(paced["answered"]), int(paced["answered"]) > 20)
	_assert_eq("paced responses without NaN", paced["nan"], 0)

	print("ROUNDTRIP unpaced60 p50_ms=%.3f p99_ms=%.3f max_ms=%.3f" % [a["rtt_p50_ms"], a["rtt_p99_ms"], a["rtt_max_ms"]])
	print("ROUNDTRIP unpaced90 p50_ms=%.3f p99_ms=%.3f max_ms=%.3f" % [b["rtt_p50_ms"], b["rtt_p99_ms"], b["rtt_max_ms"]])
	_print_paced(paced)
	_write_report()
	_finish()


# --- phases ---------------------------------------------------------------------------------

func _replay(label: String, stream: PackedFloat32Array, times, n: int, hz: float, exp_pos: PackedFloat32Array, exp_quat: PackedFloat32Array, tol_pos: float, tol_rot: float) -> Dictionary:
	var rtts: Array = []
	var max_pos := 0.0
	var max_rot := 0.0
	var nan_count := 0
	var answered := 0
	var first_pos := []  # stored for the reset test
	for k in range(n):
		var t := (float(times[k]) if times != null else float(k) / hz)
		var message := _request(stream, k, t)
		var r := _roundtrip(message)
		if r.is_empty():
			_failures.append("%s: no response for frame %d" % [label, k])
			break
		rtts.append(r["usec"])
		var frame: Dictionary = r["message"]["estimated_frame"]
		var cmp := _compare(frame, exp_pos, exp_quat, k)
		max_pos = maxf(max_pos, cmp["pos"])
		max_rot = maxf(max_rot, cmp["rot"])
		nan_count += cmp["nan"]
		answered += 1
		if k < 200:
			first_pos.append(cmp["got_pos"])
	_assert_eq("%s: all frames answered" % label, answered, n)
	_assert_eq("%s: NaN values" % label, nan_count, 0)
	_assert_true("%s: max position error %s m <= %s" % [label, String.num_scientific(max_pos), String.num_scientific(tol_pos)], max_pos <= tol_pos)
	_assert_true("%s: max rotation error %.4f deg <= %.2f" % [label, max_rot, tol_rot], max_rot <= tol_rot)
	print("PARITY %s frames=%d max_pos_m=%s max_rot_deg=%.5f" % [label, answered, String.num_scientific(max_pos), max_rot])
	var s := _stats(rtts)
	s["frames"] = answered
	s["max_pos_m"] = max_pos
	s["max_rot_deg"] = max_rot
	s["first_pos"] = first_pos
	return s


func _test_reset_reproducibility(a: Dictionary) -> void:
	var ref: Array = a["first_pos"]
	var count := ref.size()
	# 1) explicit reset_estimator message, then the same frames again
	var reset := {"format": StreamClient.CAPTURE_FORMAT, "type": "reset_estimator", "session_id": _session, "request_id": _next_id()}
	var rr := _roundtrip(reset, "estimator_reset")
	_assert_true("reset_estimator is acknowledged", not rr.is_empty())
	var worst1 := _replay_diff(ref, count)
	# 2) implicit reset: the previous replay ended at t ~ 3.3 s, this one restarts at t = 0
	var worst2 := _replay_diff(ref, count)
	print("RESET explicit_max_diff_m=%s implicit_max_diff_m=%s" % [String.num_scientific(worst1), String.num_scientific(worst2)])
	_assert_true("reset reproducibility (explicit) %s" % String.num_scientific(worst1), worst1 <= 1e-9)
	_assert_true("reset reproducibility (implicit, timestamp restart) %s" % String.num_scientific(worst2), worst2 <= 1e-9)
	_report["reset"] = {"explicit": worst1, "implicit": worst2}


func _replay_diff(ref: Array, count: int) -> float:
	var worst := 0.0
	for k in range(count):
		var r := _roundtrip(_request(_stream60, k, float(k) / 60.0))
		if r.is_empty():
			_failures.append("reset replay: no response for frame %d" % k)
			return INF
		var got := _positions(r["message"]["estimated_frame"])
		var prev: Array = ref[k]
		for i in range(got.size()):
			worst = maxf(worst, (got[i] - prev[i]).length())
	return worst


func _print_paced(paced: Dictionary) -> void:
	print("ROUNDTRIP paced%.0f p50_ms=%.3f p99_ms=%.3f max_ms=%.3f answered=%d skipped_ticks=%d of %d achieved_hz=%.1f age_at_render_p50_ms=%.2f age_at_render_p99_ms=%.2f" % [
		paced["tick_hz"], paced["rtt_p50_ms"], paced["rtt_p99_ms"], paced["rtt_max_ms"], paced["answered"], paced["skipped_ticks"], paced["ticks"],
		paced["achieved_hz"], paced["age_p50_ms"], paced["age_p99_ms"]])


func _paced(seconds: float) -> Dictionary:
	var hz := 1000000.0 / float(_tick_us)
	var ticks := 0
	ticks = mini(int(seconds * hz), int(_meta["frames90"]))
	var start := Time.get_ticks_usec()
	var rtts: Array = []
	var ages: Array = []
	var skipped := 0
	var nan_count := 0
	var in_flight := false
	var send_us := 0
	var due_of_request := 0
	var deadline := 0
	var k := 0
	# a "render loop" at 90 Hz: on every tick send the newest tracker frame unless the previous
	# request is still in flight (stop-and-wait); responses are consumed whenever they arrive
	while k < ticks or (in_flight and Time.get_ticks_usec() < deadline):
		var now := Time.get_ticks_usec()
		var res = _poll_response()
		if res != null:
			in_flight = false
			rtts.append(now - send_us)
			var tick_after := start + (int((now - start) / _tick_us) + 1) * _tick_us
			ages.append(tick_after - due_of_request)  # tracker sample -> first render tick that can show it
			nan_count += _count_nan(res["estimated_frame"])
		if k < ticks and now >= start + k * _tick_us:
			if in_flight:
				skipped += 1
			else:
				var fi := mini(int(round(float(k) * 90.0 / hz)), int(_meta["frames90"]) - 1)
				var message := _request(_stream90, fi, float(_t90[fi]))
				send_us = Time.get_ticks_usec()
				due_of_request = start + k * _tick_us
				if _client.send_message(message):
					in_flight = true
			k += 1
			if k == ticks:
				deadline = Time.get_ticks_usec() + 500000
		else:
			OS.delay_usec(25)
	var s := _stats(rtts)
	var ag := _stats(ages)
	return {
		"ticks": ticks,
		"answered": rtts.size(),
		"skipped_ticks": skipped,
		"achieved_hz": float(rtts.size()) / (float(ticks) / hz),
		"tick_hz": hz,
		"rtt_p50_ms": s["rtt_p50_ms"],
		"rtt_p99_ms": s["rtt_p99_ms"],
		"rtt_max_ms": s["rtt_max_ms"],
		"age_p50_ms": ag["rtt_p50_ms"],
		"age_p99_ms": ag["rtt_p99_ms"],
		"nan": nan_count,
	}


# --- protocol helpers -----------------------------------------------------------------------

func _next_id() -> int:
	_request_id += 1
	return _request_id


func _tracker(role: String, stream: PackedFloat32Array, base: int) -> Dictionary:
	return {
		"role": role,
		"confidence": 1.0,
		"valid": true,
		"transform": {
			"position": [stream[base], stream[base + 1], stream[base + 2]],
			"rotation": [stream[base + 3], stream[base + 4], stream[base + 5], stream[base + 6]],
		},
	}


func _request(stream: PackedFloat32Array, k: int, t: float) -> Dictionary:
	var o := k * RAW_DIM
	var sample := {
		"timestamp_seconds": t,
		"hmd": _tracker("head", stream, o),
		"left_hand": _tracker("left_hand", stream, o + 7),
		"right_hand": _tracker("right_hand", stream, o + 14),
		"optional_trackers": {},
	}
	var id := _next_id()
	return _client.build_estimate_request_message(_session, id, k, t, sample, _calibration, {}, "standing")


func _roundtrip(message: Dictionary, expect_type := "estimate_response") -> Dictionary:
	var t0 := Time.get_ticks_usec()
	if not _client.send_message(message):
		_failures.append("send_message failed")
		return {}
	while Time.get_ticks_usec() - t0 < 5000000:
		for m in _client.poll_messages():
			var type := str(m.get("type", ""))
			if type == expect_type:
				return {"message": m, "usec": Time.get_ticks_usec() - t0}
			if type == "estimate_error":
				_failures.append("estimate_error: %s" % m.get("error", ""))
				return {}
		OS.delay_usec(20)
	_failures.append("timeout waiting for %s" % expect_type)
	return {}


func _poll_response():
	for m in _client.poll_messages():
		var type := str(m.get("type", ""))
		if type == "estimate_response":
			return m
		if type == "estimate_error":
			_failures.append("estimate_error: %s" % m.get("error", ""))
	return null


func _connect() -> bool:
	var deadline := Time.get_ticks_msec() + 8000
	while Time.get_ticks_msec() < deadline:
		_client.poll_connection()
		if _client.is_stream_connected():
			return true
		OS.delay_msec(20)
	return false


# --- comparison -----------------------------------------------------------------------------

func _positions(frame: Dictionary) -> Array:
	var out: Array = []
	var joints: Dictionary = frame["joints"]
	for name in _joints:
		var p = joints[name]["transform"]["position"]
		out.append(Vector3(p[0], p[1], p[2]))
	return out


func _compare(frame: Dictionary, exp_pos: PackedFloat32Array, exp_quat: PackedFloat32Array, k: int) -> Dictionary:
	var joints: Dictionary = frame["joints"]
	var max_pos := 0.0
	var max_rot := 0.0
	var nan_count := 0
	var got_pos: Array = []
	for j in range(_nj):
		var tr: Dictionary = joints[_joints[j]]["transform"]
		var p = tr["position"]
		var q = tr["rotation"]
		var gp := Vector3(p[0], p[1], p[2])
		got_pos.append(gp)
		var e := Vector3(exp_pos[(k * _nj + j) * 3], exp_pos[(k * _nj + j) * 3 + 1], exp_pos[(k * _nj + j) * 3 + 2])
		if is_nan(gp.x) or is_nan(gp.y) or is_nan(gp.z) or is_nan(q[0]) or is_nan(q[3]):
			nan_count += 1
			continue
		max_pos = maxf(max_pos, _max_abs(gp - e))
		var eb := (k * _nj + j) * 4
		var qe := Quaternion(exp_quat[eb], exp_quat[eb + 1], exp_quat[eb + 2], exp_quat[eb + 3]).normalized()
		var qg := Quaternion(q[0], q[1], q[2], q[3]).normalized()
		var d := qe.inverse() * qg
		var ang := 2.0 * atan2(Vector3(d.x, d.y, d.z).length(), absf(d.w))
		max_rot = maxf(max_rot, rad_to_deg(ang))
	return {"pos": max_pos, "rot": max_rot, "nan": nan_count, "got_pos": got_pos}


func _max_abs(v: Vector3) -> float:
	return maxf(maxf(absf(v.x), absf(v.y)), absf(v.z))


func _count_nan(frame: Dictionary) -> int:
	var n := 0
	for name in frame["joints"].keys():
		var p = frame["joints"][name]["transform"]["position"]
		if is_nan(p[0]) or is_nan(p[1]) or is_nan(p[2]):
			n += 1
	return n


func _stats(usec: Array) -> Dictionary:
	if usec.is_empty():
		return {"rtt_p50_ms": INF, "rtt_p99_ms": INF, "rtt_max_ms": INF, "rtt_mean_ms": INF}
	var a := usec.duplicate()
	a.sort()
	var sum := 0.0
	for v in a:
		sum += float(v)
	return {
		"rtt_p50_ms": float(a[int(a.size() * 0.50)]) / 1000.0,
		"rtt_p99_ms": float(a[mini(a.size() - 1, int(a.size() * 0.99))]) / 1000.0,
		"rtt_max_ms": float(a[a.size() - 1]) / 1000.0,
		"rtt_mean_ms": sum / a.size() / 1000.0,
	}


# --- fixture / args -------------------------------------------------------------------------

func _parse_args() -> void:
	for a in OS.get_cmdline_user_args():
		var s := str(a)
		if s.begins_with("--") and s.contains("="):
			var kv := s.substr(2).split("=", true, 1)
			_args[kv[0]] = kv[1]


func _read_f32(path: String) -> PackedFloat32Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		_failures.append("cannot open %s" % path)
		return PackedFloat32Array()
	return f.get_buffer(f.get_length()).to_float32_array()


func _load_fixture() -> bool:
	var dir := str(_args.get("fixture", ""))
	if dir.is_empty():
		_failures.append("missing --fixture=DIR (see validation.sprint.parity fixture)")
		return false
	var mf := FileAccess.open(dir + "/meta.json", FileAccess.READ)
	if mf == null:
		_failures.append("cannot open %s/meta.json" % dir)
		return false
	_meta = JSON.parse_string(mf.get_as_text())
	_joints = _meta["joints"]
	_nj = int(_meta["nj"])
	_stream60 = _read_f32(dir + "/stream60.bin")
	_exp_pos60 = _read_f32(dir + "/expected_pos60.bin")
	_exp_quat60 = _read_f32(dir + "/expected_quat60.bin")
	_stream90 = _read_f32(dir + "/stream90.bin")
	_exp_pos90 = _read_f32(dir + "/expected_pos90.bin")
	_exp_quat90 = _read_f32(dir + "/expected_quat90.bin")
	var tf := FileAccess.open(dir + "/t90.bin", FileAccess.READ)
	_t90 = tf.get_buffer(tf.get_length()).to_float64_array()
	var n60 := int(_meta["frames60"])
	var n90 := int(_meta["frames90"])
	if _stream60.size() != n60 * RAW_DIM or _stream90.size() != n90 * RAW_DIM or _t90.size() != n90:
		_failures.append("fixture sizes do not match meta.json")
		return false
	if _exp_pos60.size() != n60 * _nj * 3 or _exp_quat60.size() != n60 * _nj * 4:
		_failures.append("expected_*60 sizes do not match meta.json")
		return false
	var bl: Dictionary = _meta["bone_lengths_m"].duplicate()
	bl["arm_length"] = _meta["calib"]["arm_length_m"]
	bl["shoulder_width"] = _meta["calib"]["shoulder_width_m"]
	_calibration = {"human_height_m": _meta["calib"]["height_m"], "avatar_scale": 1.0, "bone_lengths_m": bl}
	return true


func _write_report() -> void:
	var path := str(_args.get("report", ""))
	if path.is_empty():
		return
	var out := _report.duplicate(true)
	for key in ["A_60hz", "B_90hz"]:
		if out.has(key):
			out[key].erase("first_pos")
	var f := FileAccess.open(path, FileAccess.WRITE)
	if f != null:
		f.store_string(JSON.stringify(out, "  "))


# --- asserts --------------------------------------------------------------------------------

func _assert_true(label: String, value: bool) -> void:
	if not value:
		_failures.append("%s expected true" % label)


func _assert_eq(label: String, actual, expected) -> void:
	if actual != expected:
		_failures.append("%s actual=%s expected=%s" % [label, actual, expected])


func _finish() -> void:
	if _failures.is_empty():
		print("Godot model parity tests passed")
		quit(0)
		return
	for failure in _failures:
		push_error(failure)
	quit(1)

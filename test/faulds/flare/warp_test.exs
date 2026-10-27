defmodule Faulds.Flare.WarpTest do
  use ExUnit.Case, async: true

  alias Faulds.Flare.Warp

  # Real-image, real-pose cross-checks against FLARE's own Python
  # (`datasets.FPdataset.FingerPoseEvalDataset`/`Descdataset`) live in
  # when this port was written:
  # a pure-translation pose-stage warp matched bit-exactly, and a
  # scale+rotate+translate descriptor-stage warp (matching `matrix`/
  # `odx`/`ody`'s convention, verified empirically against `cv2` — see
  # this module's own moduledoc) matched to ~3.8e-5 max abs diff on a
  # [-1, 1]-scale image. This file covers the arithmetic in isolation;
  # `flare_nx_regression_test.exs` covers the full pipeline end-to-end.

  describe "affine_matrix/4" do
    test "pure translation (identity rotation): tx/ty is just trans + trans_2" do
      t = Warp.affine_matrix(1.0, 0.0, {-10.0, -20.0}, {5.0, 5.0})

      assert_in_delta t.a, 1.0, 1.0e-9
      assert_in_delta t.b, 0.0, 1.0e-9
      assert_in_delta t.c, 0.0, 1.0e-9
      assert_in_delta t.d, 1.0, 1.0e-9
      assert_in_delta t.tx, -5.0, 1.0e-9
      assert_in_delta t.ty, -15.0, 1.0e-9
    end

    test "matches an independent re-derivation of R @ trans + trans_2 with scale and rotation" do
      scale = 0.7
      theta = :math.pi() / 6
      trans = {-3.0, 8.0}
      trans_2 = {12.0, -4.0}

      t = Warp.affine_matrix(scale, theta, trans, trans_2)

      cos_t = :math.cos(theta)
      sin_t = :math.sin(theta)
      {tx0, ty0} = trans
      {t2x, t2y} = trans_2

      expected_a = cos_t * scale
      expected_b = -sin_t * scale
      expected_c = sin_t * scale
      expected_d = cos_t * scale
      expected_tx = expected_a * tx0 + expected_b * ty0 + t2x
      expected_ty = expected_c * tx0 + expected_d * ty0 + t2y

      assert_in_delta t.a, expected_a, 1.0e-9
      assert_in_delta t.b, expected_b, 1.0e-9
      assert_in_delta t.c, expected_c, 1.0e-9
      assert_in_delta t.d, expected_d, 1.0e-9
      assert_in_delta t.tx, expected_tx, 1.0e-9
      assert_in_delta t.ty, expected_ty, 1.0e-9
    end
  end

  describe "invert/1 and apply_point/2" do
    test "inverting a pure translation negates tx/ty and leaves the linear part alone" do
      t = Warp.affine_matrix(1.0, 0.0, {-194.0, -187.0}, {256.0, 256.0})
      inv = Warp.invert(t)

      assert_in_delta inv.a, 1.0, 1.0e-9
      assert_in_delta inv.b, 0.0, 1.0e-9
      assert_in_delta inv.c, 0.0, 1.0e-9
      assert_in_delta inv.d, 1.0, 1.0e-9
      assert_in_delta inv.tx, -t.tx, 1.0e-9
      assert_in_delta inv.ty, -t.ty, 1.0e-9
    end

    test "invert/1 undoes apply_point/2 (round-trip) for a general affine map" do
      t = Warp.affine_matrix(0.6, :math.pi() / 5, {-13.0, 27.0}, {50.0, 40.0})
      point = {17.5, -3.25}

      warped = Warp.apply_point(t, point)
      restored = Warp.apply_point(Warp.invert(t), warped)

      {x, y} = point
      {rx, ry} = restored
      assert_in_delta rx, x, 1.0e-6
      assert_in_delta ry, y, 1.0e-6
    end

    test "matches FLARE's own T_inv correction on a real pose-stage T" do
      # T for a 388 x 374 FVC2002 print's pose-stage canvas warp, recorded
      # when this port was checked against FLARE's Python: img_c=(194,187),
      # center=(256,256) -> tx=62, ty=69, no rotation/scale.
      t = Warp.affine_matrix(1.0, 0.0, {-194.0, -187.0}, {256.0, 256.0})
      # A real pose_2d this project's pose.onnx predicted on that same
      # canvas (verify_pose_export.py, during this port's development).
      raw_pose = {232.96265, 305.78937}

      corrected = Warp.apply_point(Warp.invert(t), raw_pose)

      # Undoing a pure +62/+69 translation is just subtracting it back out.
      {rx, ry} = raw_pose
      assert_in_delta elem(corrected, 0), rx - 62.0, 1.0e-6
      assert_in_delta elem(corrected, 1), ry - 69.0, 1.0e-6
    end
  end

  describe "warp/4" do
    test "pure integer translation moves content by exactly (tx, ty), bilinear-exact" do
      # 4x4 image, single bright pixel at (row=1, col=1).
      rows =
        for row <- 0..3 do
          for col <- 0..3, do: if(row == 1 and col == 1, do: 100.0, else: 0.0)
        end

      image = Nx.tensor(rows, type: :f32)
      t = Warp.affine_matrix(1.0, 0.0, {0.0, 0.0}, {2.0, 1.0})

      warped = Warp.warp(image, t, {4, 4}, background: 0.0)

      expected_rows =
        for row <- 0..3 do
          for col <- 0..3, do: if(row == 2 and col == 3, do: 100.0, else: 0.0)
        end

      expected = Nx.tensor(expected_rows, type: :f32)

      max_abs_diff =
        warped |> Nx.subtract(expected) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

      assert max_abs_diff < 1.0e-4
    end

    test "out-of-bounds destination pixels are filled with :background" do
      image = Nx.broadcast(10.0, {4, 4})
      # Shift far enough that the whole 4x4 canvas samples outside the source.
      t = Warp.affine_matrix(1.0, 0.0, {0.0, 0.0}, {100.0, 100.0})

      warped = Warp.warp(image, t, {4, 4}, background: 255.0)

      assert warped |> Nx.equal(255.0) |> Nx.all() |> Nx.to_number() == 1
    end

    test "defaults background to 0.0" do
      image = Nx.broadcast(10.0, {4, 4})
      t = Warp.affine_matrix(1.0, 0.0, {0.0, 0.0}, {100.0, 100.0})

      warped = Warp.warp(image, t, {4, 4})

      assert warped |> Nx.equal(0.0) |> Nx.all() |> Nx.to_number() == 1
    end

    test "output has the requested target shape regardless of source shape" do
      image = Nx.broadcast(1.0, {10, 6})
      t = Warp.affine_matrix(1.0, 0.0, {0.0, 0.0}, {0.0, 0.0})

      warped = Warp.warp(image, t, {8, 8})

      assert Nx.shape(warped) == {8, 8}
    end
  end
end

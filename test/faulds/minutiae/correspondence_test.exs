defmodule Faulds.Minutiae.CorrespondenceTest do
  use ExUnit.Case, async: true

  alias Faulds.Minutiae.Correspondence

  describe "match/3 on synthetic minutiae" do
    setup do
      :rand.seed(:exsplus, {42, 1337, 7})

      base =
        for _ <- 1..40 do
          %{
            x: :rand.uniform(400) * 1.0,
            y: :rand.uniform(400) * 1.0,
            angle: :rand.uniform(360) * 1.0,
            quality: 50 + :rand.uniform(50)
          }
        end

      {:ok, base: base}
    end

    test "recovers correspondences under a pure translation", %{base: base} do
      shifted = Enum.map(base, &%{&1 | x: &1.x + 15.0, y: &1.y - 8.0})

      %{pairs: pairs, transform: transform} = Correspondence.match(base, shifted)

      assert length(pairs) >= round(length(base) * 0.9)
      assert_in_delta transform.rotation, 0.0, 1.0
      assert_in_delta transform.dx, 15.0, 1.0
      assert_in_delta transform.dy, -8.0, 1.0
      assert one_to_one?(pairs)
    end

    test "recovers correspondences under rotation + translation", %{base: base} do
      rotated =
        Enum.map(base, fn m ->
          radians = 30.0 * :math.pi() / 180.0
          rx = m.x * :math.cos(radians) - m.y * :math.sin(radians)
          ry = m.x * :math.sin(radians) + m.y * :math.cos(radians)
          %{m | x: rx + 50.0, y: ry + 20.0, angle: :math.fmod(m.angle + 30.0, 360.0)}
        end)

      %{pairs: pairs, transform: transform} = Correspondence.match(base, rotated)

      assert length(pairs) >= round(length(base) * 0.9)
      assert_in_delta transform.rotation, 30.0, 1.0
      assert one_to_one?(pairs)
    end

    test "tolerates small positional jitter", %{base: base} do
      :rand.seed(:exsplus, {9, 9, 9})

      jittered =
        Enum.map(base, fn m ->
          %{m | x: m.x + 3.0 + :rand.uniform() * 2, y: m.y - 3.0 + :rand.uniform() * 2}
        end)

      %{pairs: pairs} = Correspondence.match(base, jittered)

      assert length(pairs) >= round(length(base) * 0.7)
      assert one_to_one?(pairs)
    end

    test "finds few or no matches between unrelated point sets", %{base: base} do
      :rand.seed(:exsplus, {123, 456, 789})

      unrelated =
        for _ <- 1..40 do
          %{
            x: :rand.uniform(400) * 1.0,
            y: :rand.uniform(400) * 1.0,
            angle: :rand.uniform(360) * 1.0,
            quality: 50 + :rand.uniform(50)
          }
        end

      %{pairs: pairs} = Correspondence.match(base, unrelated)

      assert length(pairs) <= round(length(base) * 0.25)
    end

    test "empty input yields no matches and no transform", %{base: base} do
      assert Correspondence.match([], base) == %{pairs: [], transform: nil}
      assert Correspondence.match(base, []) == %{pairs: [], transform: nil}
      assert Correspondence.match([], []) == %{pairs: [], transform: nil}
    end
  end

  defp one_to_one?(pairs) do
    a_indices = Enum.map(pairs, &elem(&1, 0))
    b_indices = Enum.map(pairs, &elem(&1, 1))
    Enum.uniq(a_indices) == a_indices and Enum.uniq(b_indices) == b_indices
  end
end

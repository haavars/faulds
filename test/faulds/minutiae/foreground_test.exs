defmodule Faulds.Minutiae.ForegroundTest do
  use ExUnit.Case, async: true

  alias Faulds.Minutiae.Foreground

  # Foreground only in the crop's top-left 8x8 cells.
  @mask for(row <- 0..15, col <- 0..15, do: if(row < 8 and col < 8, do: 1.0, else: 0.0))

  test "mask_value/3: an upright crop centred on the pose shows the image at half scale" do
    pose = {300.0, 300.0, 0.0}
    # Image (300, 300) is the crop's centre (128, 128): cell (8, 8), background.
    assert Foreground.mask_value(@mask, {300.0, 300.0}, pose) == 0.0
    # 100 px up and left is crop (78, 78): cell (4, 4), foreground.
    assert Foreground.mask_value(@mask, {200.0, 200.0}, pose) == 1.0
    # Outside the crop (more than 256 px away at half scale).
    assert Foreground.mask_value(@mask, {0.0, 0.0}, pose) == 0.0
  end

  test "mask_value/3 follows the pose's rotation" do
    # Rotated 90°: crop = 0.5 R(90)(p - c) + 128, so image (400, 200) maps to
    # crop (128 + 50, 128 + 50) -> background, and image (200, 400) to (78, 78).
    pose = {300.0, 300.0, 90.0}
    assert Foreground.mask_value(@mask, {200.0, 400.0}, pose) == 1.0
    assert Foreground.mask_value(@mask, {400.0, 200.0}, pose) == 0.0
  end

  test "filter_xyt/3 keeps rows by quality and mask, in mindtct's bottom-left frame" do
    height = 600
    # Image (200, 200) is .xyt y = 400; image (300, 300) is y = 300.
    xyt = "200 400 90 50\n300 300 45 50\n200 400 10 5\n"

    assert Foreground.filter_xyt(xyt, height, min_quality: 20) == "200 400 90 50\n300 300 45 50\n"

    assert Foreground.filter_xyt(xyt, height,
             mask_threshold: 0.5,
             mask: @mask,
             pose: {300.0, 300.0, 0.0}
           ) == "200 400 90 50\n200 400 10 5\n"

    assert Foreground.filter_xyt(xyt, height,
             min_quality: 20,
             mask_threshold: 0.5,
             mask: @mask,
             pose: {300.0, 300.0, 0.0}
           ) == "200 400 90 50\n"
  end
end

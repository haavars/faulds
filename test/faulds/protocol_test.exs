defmodule Faulds.ProtocolTest do
  use ExUnit.Case, async: true

  alias Faulds.Protocol
  alias Faulds.Protocol.Sd302

  # Two impressions of subject a's finger 1, one of its finger 2, one of b's.
  defp protocol(listed) do
    %Protocol{
      name: "test",
      probes: [
        %{id: "p1", subject: "a", finger: 1},
        %{id: "p2", subject: "b", finger: 3},
        %{id: "p3", subject: "a", finger: nil}
      ],
      gallery: [
        %{id: "a1", subject: "a", finger: 1},
        %{id: "a1'", subject: "a", finger: 1},
        %{id: "a2", subject: "a", finger: 2},
        %{id: "b3", subject: "b", finger: 3}
      ],
      listed: listed
    }
  end

  describe "mates/2" do
    test "credits a finger's every impression, or the subject's every print" do
      p = protocol(nil)
      assert Protocol.mates(p) == [[0, 1], [3], []]
      assert Protocol.mates(p, :subject) == [[0, 1, 2], [3], [0, 1, 2]]
    end

    test "a listed protocol credits its list by default" do
      p = protocol(%{"p1" => ["a1'"], "p2" => ["b3"]})
      assert Protocol.mates(p) == [[1], [3], []]
      assert Protocol.mates(p, :finger) == [[0, 1], [3], []]
    end
  end

  test "unlisted_mates/1 finds other impressions of a probe's finger left off its list" do
    assert Protocol.unlisted_mates(protocol(%{"p1" => ["a1'"], "p2" => ["b3"]})) == [{"p1", "a1"}]
    assert Protocol.unlisted_mates(protocol(nil)) == []
  end

  test "with_distractors/3 appends, keeping columns, and refuses a duplicate" do
    p = Protocol.with_distractors(protocol(nil), [%{id: "d", subject: "z", finger: 1}], "G+d")
    assert p.name == "G+d" and Protocol.mates(p) == [[0, 1], [3], []]

    assert_raise ArgumentError, fn ->
      Protocol.with_distractors(p, [%{id: "a2", subject: "a", finger: 2}], "dup")
    end
  end

  describe "Sd302" do
    @e "00002302_6C_X_199_BT_D800_1106PPI_16BPC_1CH_LP09_1"
    @h "00002302_6C_X_199_BT_D800_1000PPI_8BPC_1CH_LP09_1.lffs"

    test "parses a latent name and joins SD302e and SD302h names by canonical key" do
      assert %{subject: "00002302", activity: "6C", hand: "X", encounter: "199", technique: "BT"} =
               Sd302.parse(@e)

      assert %{digitizer: "D800", ppi: 1106, bits: 16, lp: 9, source: "1"} = Sd302.parse(@e)
      assert Sd302.canonical_key(@e) == Sd302.canonical_key("some/dir/" <> @h)
      assert Sd302.canonical_key(@e) == "00002302_6C_X_199_BT_D800_LP09_1"
      assert Sd302.parse("00002302_08") == nil
    end

    @tag :tmp_dir
    test "Tsinghua's list: every finger of its subjects in the gallery, the list as mates",
         %{tmp_dir: dir} do
      list = Path.join(dir, "pairs.txt")

      File.write!(list, """
      00002302_1E_L_L01_BP_S04_1200PPI_8BPC_1CH_LP05_1,00002302_08
      00002302_6C_X_199_BT_D800_1106PPI_16BPC_1CH_LP09_1,00002302_10
      00002305_1E_R_L01_BP_S04_1200PPI_8BPC_1CH_LP02_1,00002305_07\
      """)

      p = Sd302.tsinghua(list)
      assert length(p.gallery) == 20
      assert hd(p.gallery) == %{id: "00002302_01", subject: "00002302", finger: 1}
      assert Protocol.mates(p) == [[7], [9], [16]]
      assert Protocol.mates(p, :finger) == Protocol.mates(p)
      assert Protocol.unlisted_mates(p) == []
      # Right hand, left finger.
      assert Sd302.hand_mismatches(p) == ["00002305_1E_R_L01_BP_S04_1200PPI_8BPC_1CH_LP02_1"]
    end

    @tag :tmp_dir
    test "the errata: reasons by bullet, the old name of a moved latent, the corrected list",
         %{tmp_dir: dir} do
      errata = Path.join(dir, "errata.txt")

      File.write!(errata, """
      ## 18 June 2024

       * 00002361, Device V, right ring is actually right index.
       * 00002480_6E_X_050_BT_D800_1103PPI_16BPC_1CH_LP20_1 source is finger 3, but is labeled as finger 2.

      ## 19 July 2022
      * These latents were identified as possible palm/joint prints:

      00002302_6C_X_199_BT_D800_1106PPI_16BPC_1CH_LP09_1.png

      ## 16 December 2019
       * Three latents from identifier 00002614 were mislabeled. These three latents
         belong to subject 00002574:
          - 00002614_6C_X_214_BT_D800_1000PPI_8BPC_1CH_LP16_1.lffs change to 00002574_6C_X_214_BT_D800_1000PPI_8BPC_1CH_LP16_1.lffs
       * The dimensions for the latent image 00002302_6C_X_199_BT_D800_1106PPI_16BPC_1CH_LP09_1.png have changed slightly.
       * The dimensions for the latent image 00002302_1E_L_L01_BP_S04_1200PPI_8BPC_1CH_LP05_1.png have changed slightly.
      """)

      flags = Sd302.errata(errata)

      assert flags == %{
               "00002480_6E_X_050_BT_D800_LP20_1" => :finger,
               "00002302_6C_X_199_BT_D800_LP09_1" => :palm_or_joint,
               "00002614_6C_X_214_BT_D800_LP16_1" => :moved,
               "00002302_1E_L_L01_BP_S04_LP05_1" => :dimensions
             }

      p = %Protocol{
        name: "N2N (published list)",
        probes: [
          %{
            id: "00002302_1E_L_L01_BP_S04_1200PPI_8BPC_1CH_LP05_1",
            subject: "00002302",
            finger: 8
          },
          %{
            id: "00002302_6C_X_199_BT_D800_1106PPI_16BPC_1CH_LP09_1",
            subject: "00002302",
            finger: 10
          }
        ],
        gallery: [],
        listed: %{}
      }

      corrected = Sd302.corrected(p, flags)
      assert corrected.name == "N2N (corrected list)"
      assert Enum.map(corrected.probes, & &1.finger) == [8]
    end
  end
end

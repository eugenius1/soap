--------------------------------------------------------------------------------
--                               Compressor_3_2
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Popa, Illyes Kinga, 2012
--------------------------------------------------------------------------------
-- combinatorial

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity Compressor_3_2 is
   port ( X0 : in  std_logic_vector(2 downto 0);
          R : out  std_logic_vector(1 downto 0)   );
end entity;

architecture arch of Compressor_3_2 is
signal X :  std_logic_vector(2 downto 0);
begin
   X <=X0 ;
   with X select R <= 
      "00" when "000", 
      "01" when "001", 
      "01" when "010", 
      "10" when "011", 
      "01" when "100", 
      "10" when "101", 
      "10" when "110", 
      "11" when "111", 
      "--" when others;

end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_42_f400_uid14
--                     (IntAdderClassical_42_f400_uid16)
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2010)
--------------------------------------------------------------------------------
-- Pipeline depth: 0 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_42_f400_uid14 is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(41 downto 0);
          Y : in  std_logic_vector(41 downto 0);
          Cin : in std_logic;
          R : out  std_logic_vector(41 downto 0)   );
end entity;

architecture arch of IntAdder_42_f400_uid14 is
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
         end if;
      end process;
   --Classical
    R <= X + Y + Cin;
end architecture;

--------------------------------------------------------------------------------
--                IntMultiplier_UsingDSP_24_24_0_unsigned_uid3
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Kinga Illyes, Bogdan Popa, Bogdan Pasca, 2012
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntMultiplier_UsingDSP_24_24_0_unsigned_uid3 is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(23 downto 0);
          Y : in  std_logic_vector(23 downto 0);
          R : out  std_logic_vector(47 downto 0)   );
end entity;

architecture arch of IntMultiplier_UsingDSP_24_24_0_unsigned_uid3 is
   component IntAdder_42_f400_uid14 is
      port ( clk, rst : in std_logic;
             X : in  std_logic_vector(41 downto 0);
             Y : in  std_logic_vector(41 downto 0);
             Cin : in std_logic;
             R : out  std_logic_vector(41 downto 0)   );
   end component;

signal XX_m4 :  std_logic_vector(23 downto 0);
signal YY_m4 :  std_logic_vector(23 downto 0);
signal DSP_bh5_ch0_0 :  std_logic_vector(40 downto 0);
signal heap_bh5_w47_0, heap_bh5_w47_0_d1 : std_logic;
signal heap_bh5_w46_0, heap_bh5_w46_0_d1 : std_logic;
signal heap_bh5_w45_0, heap_bh5_w45_0_d1 : std_logic;
signal heap_bh5_w44_0, heap_bh5_w44_0_d1 : std_logic;
signal heap_bh5_w43_0, heap_bh5_w43_0_d1 : std_logic;
signal heap_bh5_w42_0, heap_bh5_w42_0_d1 : std_logic;
signal heap_bh5_w41_0, heap_bh5_w41_0_d1 : std_logic;
signal heap_bh5_w40_0, heap_bh5_w40_0_d1 : std_logic;
signal heap_bh5_w39_0, heap_bh5_w39_0_d1 : std_logic;
signal heap_bh5_w38_0, heap_bh5_w38_0_d1 : std_logic;
signal heap_bh5_w37_0, heap_bh5_w37_0_d1 : std_logic;
signal heap_bh5_w36_0, heap_bh5_w36_0_d1 : std_logic;
signal heap_bh5_w35_0, heap_bh5_w35_0_d1 : std_logic;
signal heap_bh5_w34_0, heap_bh5_w34_0_d1 : std_logic;
signal heap_bh5_w33_0, heap_bh5_w33_0_d1 : std_logic;
signal heap_bh5_w32_0, heap_bh5_w32_0_d1 : std_logic;
signal heap_bh5_w31_0, heap_bh5_w31_0_d1 : std_logic;
signal heap_bh5_w30_0, heap_bh5_w30_0_d1 : std_logic;
signal heap_bh5_w29_0, heap_bh5_w29_0_d1 : std_logic;
signal heap_bh5_w28_0, heap_bh5_w28_0_d1 : std_logic;
signal heap_bh5_w27_0, heap_bh5_w27_0_d1 : std_logic;
signal heap_bh5_w26_0, heap_bh5_w26_0_d1 : std_logic;
signal heap_bh5_w25_0, heap_bh5_w25_0_d1 : std_logic;
signal heap_bh5_w24_0, heap_bh5_w24_0_d1 : std_logic;
signal heap_bh5_w23_0, heap_bh5_w23_0_d1 : std_logic;
signal heap_bh5_w22_0, heap_bh5_w22_0_d1 : std_logic;
signal heap_bh5_w21_0, heap_bh5_w21_0_d1 : std_logic;
signal heap_bh5_w20_0, heap_bh5_w20_0_d1 : std_logic;
signal heap_bh5_w19_0, heap_bh5_w19_0_d1 : std_logic;
signal heap_bh5_w18_0, heap_bh5_w18_0_d1 : std_logic;
signal heap_bh5_w17_0, heap_bh5_w17_0_d1 : std_logic;
signal heap_bh5_w16_0, heap_bh5_w16_0_d1 : std_logic;
signal heap_bh5_w15_0, heap_bh5_w15_0_d1 : std_logic;
signal heap_bh5_w14_0, heap_bh5_w14_0_d1 : std_logic;
signal heap_bh5_w13_0, heap_bh5_w13_0_d1 : std_logic;
signal heap_bh5_w12_0, heap_bh5_w12_0_d1 : std_logic;
signal heap_bh5_w11_0, heap_bh5_w11_0_d1 : std_logic;
signal heap_bh5_w10_0, heap_bh5_w10_0_d1 : std_logic;
signal heap_bh5_w9_0, heap_bh5_w9_0_d1 : std_logic;
signal heap_bh5_w8_0, heap_bh5_w8_0_d1 : std_logic;
signal heap_bh5_w7_0, heap_bh5_w7_0_d1 : std_logic;
signal DSP_bh5_ch1_0 :  std_logic_vector(40 downto 0);
signal heap_bh5_w30_1, heap_bh5_w30_1_d1 : std_logic;
signal heap_bh5_w29_1, heap_bh5_w29_1_d1 : std_logic;
signal heap_bh5_w28_1, heap_bh5_w28_1_d1 : std_logic;
signal heap_bh5_w27_1, heap_bh5_w27_1_d1 : std_logic;
signal heap_bh5_w26_1, heap_bh5_w26_1_d1 : std_logic;
signal heap_bh5_w25_1, heap_bh5_w25_1_d1 : std_logic;
signal heap_bh5_w24_1, heap_bh5_w24_1_d1 : std_logic;
signal heap_bh5_w23_1, heap_bh5_w23_1_d1 : std_logic;
signal heap_bh5_w22_1, heap_bh5_w22_1_d1 : std_logic;
signal heap_bh5_w21_1, heap_bh5_w21_1_d1 : std_logic;
signal heap_bh5_w20_1, heap_bh5_w20_1_d1 : std_logic;
signal heap_bh5_w19_1, heap_bh5_w19_1_d1 : std_logic;
signal heap_bh5_w18_1, heap_bh5_w18_1_d1 : std_logic;
signal heap_bh5_w17_1, heap_bh5_w17_1_d1 : std_logic;
signal heap_bh5_w16_1, heap_bh5_w16_1_d1 : std_logic;
signal heap_bh5_w15_1, heap_bh5_w15_1_d1 : std_logic;
signal heap_bh5_w14_1, heap_bh5_w14_1_d1 : std_logic;
signal heap_bh5_w13_1, heap_bh5_w13_1_d1 : std_logic;
signal heap_bh5_w12_1, heap_bh5_w12_1_d1 : std_logic;
signal heap_bh5_w11_1, heap_bh5_w11_1_d1 : std_logic;
signal heap_bh5_w10_1, heap_bh5_w10_1_d1 : std_logic;
signal heap_bh5_w9_1, heap_bh5_w9_1_d1 : std_logic;
signal heap_bh5_w8_1, heap_bh5_w8_1_d1 : std_logic;
signal heap_bh5_w7_1, heap_bh5_w7_1_d1 : std_logic;
signal heap_bh5_w6_0 : std_logic;
signal heap_bh5_w5_0 : std_logic;
signal heap_bh5_w4_0 : std_logic;
signal heap_bh5_w3_0 : std_logic;
signal heap_bh5_w2_0 : std_logic;
signal heap_bh5_w1_0 : std_logic;
signal heap_bh5_w0_0 : std_logic;
signal finalAdderIn0_bh5 :  std_logic_vector(41 downto 0);
signal finalAdderIn1_bh5 :  std_logic_vector(41 downto 0);
signal finalAdderCin_bh5 : std_logic;
signal finalAdderOut_bh5 :  std_logic_vector(41 downto 0);
signal tempR_bh5_0, tempR_bh5_0_d1 :  std_logic_vector(6 downto 0);
signal CompressionResult5 :  std_logic_vector(48 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            heap_bh5_w47_0_d1 <=  heap_bh5_w47_0;
            heap_bh5_w46_0_d1 <=  heap_bh5_w46_0;
            heap_bh5_w45_0_d1 <=  heap_bh5_w45_0;
            heap_bh5_w44_0_d1 <=  heap_bh5_w44_0;
            heap_bh5_w43_0_d1 <=  heap_bh5_w43_0;
            heap_bh5_w42_0_d1 <=  heap_bh5_w42_0;
            heap_bh5_w41_0_d1 <=  heap_bh5_w41_0;
            heap_bh5_w40_0_d1 <=  heap_bh5_w40_0;
            heap_bh5_w39_0_d1 <=  heap_bh5_w39_0;
            heap_bh5_w38_0_d1 <=  heap_bh5_w38_0;
            heap_bh5_w37_0_d1 <=  heap_bh5_w37_0;
            heap_bh5_w36_0_d1 <=  heap_bh5_w36_0;
            heap_bh5_w35_0_d1 <=  heap_bh5_w35_0;
            heap_bh5_w34_0_d1 <=  heap_bh5_w34_0;
            heap_bh5_w33_0_d1 <=  heap_bh5_w33_0;
            heap_bh5_w32_0_d1 <=  heap_bh5_w32_0;
            heap_bh5_w31_0_d1 <=  heap_bh5_w31_0;
            heap_bh5_w30_0_d1 <=  heap_bh5_w30_0;
            heap_bh5_w29_0_d1 <=  heap_bh5_w29_0;
            heap_bh5_w28_0_d1 <=  heap_bh5_w28_0;
            heap_bh5_w27_0_d1 <=  heap_bh5_w27_0;
            heap_bh5_w26_0_d1 <=  heap_bh5_w26_0;
            heap_bh5_w25_0_d1 <=  heap_bh5_w25_0;
            heap_bh5_w24_0_d1 <=  heap_bh5_w24_0;
            heap_bh5_w23_0_d1 <=  heap_bh5_w23_0;
            heap_bh5_w22_0_d1 <=  heap_bh5_w22_0;
            heap_bh5_w21_0_d1 <=  heap_bh5_w21_0;
            heap_bh5_w20_0_d1 <=  heap_bh5_w20_0;
            heap_bh5_w19_0_d1 <=  heap_bh5_w19_0;
            heap_bh5_w18_0_d1 <=  heap_bh5_w18_0;
            heap_bh5_w17_0_d1 <=  heap_bh5_w17_0;
            heap_bh5_w16_0_d1 <=  heap_bh5_w16_0;
            heap_bh5_w15_0_d1 <=  heap_bh5_w15_0;
            heap_bh5_w14_0_d1 <=  heap_bh5_w14_0;
            heap_bh5_w13_0_d1 <=  heap_bh5_w13_0;
            heap_bh5_w12_0_d1 <=  heap_bh5_w12_0;
            heap_bh5_w11_0_d1 <=  heap_bh5_w11_0;
            heap_bh5_w10_0_d1 <=  heap_bh5_w10_0;
            heap_bh5_w9_0_d1 <=  heap_bh5_w9_0;
            heap_bh5_w8_0_d1 <=  heap_bh5_w8_0;
            heap_bh5_w7_0_d1 <=  heap_bh5_w7_0;
            heap_bh5_w30_1_d1 <=  heap_bh5_w30_1;
            heap_bh5_w29_1_d1 <=  heap_bh5_w29_1;
            heap_bh5_w28_1_d1 <=  heap_bh5_w28_1;
            heap_bh5_w27_1_d1 <=  heap_bh5_w27_1;
            heap_bh5_w26_1_d1 <=  heap_bh5_w26_1;
            heap_bh5_w25_1_d1 <=  heap_bh5_w25_1;
            heap_bh5_w24_1_d1 <=  heap_bh5_w24_1;
            heap_bh5_w23_1_d1 <=  heap_bh5_w23_1;
            heap_bh5_w22_1_d1 <=  heap_bh5_w22_1;
            heap_bh5_w21_1_d1 <=  heap_bh5_w21_1;
            heap_bh5_w20_1_d1 <=  heap_bh5_w20_1;
            heap_bh5_w19_1_d1 <=  heap_bh5_w19_1;
            heap_bh5_w18_1_d1 <=  heap_bh5_w18_1;
            heap_bh5_w17_1_d1 <=  heap_bh5_w17_1;
            heap_bh5_w16_1_d1 <=  heap_bh5_w16_1;
            heap_bh5_w15_1_d1 <=  heap_bh5_w15_1;
            heap_bh5_w14_1_d1 <=  heap_bh5_w14_1;
            heap_bh5_w13_1_d1 <=  heap_bh5_w13_1;
            heap_bh5_w12_1_d1 <=  heap_bh5_w12_1;
            heap_bh5_w11_1_d1 <=  heap_bh5_w11_1;
            heap_bh5_w10_1_d1 <=  heap_bh5_w10_1;
            heap_bh5_w9_1_d1 <=  heap_bh5_w9_1;
            heap_bh5_w8_1_d1 <=  heap_bh5_w8_1;
            heap_bh5_w7_1_d1 <=  heap_bh5_w7_1;
            tempR_bh5_0_d1 <=  tempR_bh5_0;
         end if;
      end process;
   XX_m4 <= X ;
   YY_m4 <= Y ;
   
   -- Beginning of code generated by BitHeap::generateCompressorVHDL
   -- code generated by BitHeap::generateSupertileVHDL()
   ----------------Synchro barrier, entering cycle 0----------------
   DSP_bh5_ch0_0 <= ("" & XX_m4(23 downto 0) & "") * ("" & YY_m4(23 downto 7) & "");
   heap_bh5_w47_0 <= DSP_bh5_ch0_0(40); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w46_0 <= DSP_bh5_ch0_0(39); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w45_0 <= DSP_bh5_ch0_0(38); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w44_0 <= DSP_bh5_ch0_0(37); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w43_0 <= DSP_bh5_ch0_0(36); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w42_0 <= DSP_bh5_ch0_0(35); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w41_0 <= DSP_bh5_ch0_0(34); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w40_0 <= DSP_bh5_ch0_0(33); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w39_0 <= DSP_bh5_ch0_0(32); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w38_0 <= DSP_bh5_ch0_0(31); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w37_0 <= DSP_bh5_ch0_0(30); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w36_0 <= DSP_bh5_ch0_0(29); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w35_0 <= DSP_bh5_ch0_0(28); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w34_0 <= DSP_bh5_ch0_0(27); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w33_0 <= DSP_bh5_ch0_0(26); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w32_0 <= DSP_bh5_ch0_0(25); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w31_0 <= DSP_bh5_ch0_0(24); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w30_0 <= DSP_bh5_ch0_0(23); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w29_0 <= DSP_bh5_ch0_0(22); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w28_0 <= DSP_bh5_ch0_0(21); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w27_0 <= DSP_bh5_ch0_0(20); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w26_0 <= DSP_bh5_ch0_0(19); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w25_0 <= DSP_bh5_ch0_0(18); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w24_0 <= DSP_bh5_ch0_0(17); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w23_0 <= DSP_bh5_ch0_0(16); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w22_0 <= DSP_bh5_ch0_0(15); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w21_0 <= DSP_bh5_ch0_0(14); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w20_0 <= DSP_bh5_ch0_0(13); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w19_0 <= DSP_bh5_ch0_0(12); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w18_0 <= DSP_bh5_ch0_0(11); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w17_0 <= DSP_bh5_ch0_0(10); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w16_0 <= DSP_bh5_ch0_0(9); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w15_0 <= DSP_bh5_ch0_0(8); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w14_0 <= DSP_bh5_ch0_0(7); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w13_0 <= DSP_bh5_ch0_0(6); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w12_0 <= DSP_bh5_ch0_0(5); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w11_0 <= DSP_bh5_ch0_0(4); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w10_0 <= DSP_bh5_ch0_0(3); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w9_0 <= DSP_bh5_ch0_0(2); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w8_0 <= DSP_bh5_ch0_0(1); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w7_0 <= DSP_bh5_ch0_0(0); -- cycle= 0 cp= 1.638e-09
   ----------------Synchro barrier, entering cycle 0----------------
   DSP_bh5_ch1_0 <= ("" & XX_m4(23 downto 0) & "") * ("" & YY_m4(6 downto 0) & "0000000000");
   heap_bh5_w30_1 <= DSP_bh5_ch1_0(40); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w29_1 <= DSP_bh5_ch1_0(39); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w28_1 <= DSP_bh5_ch1_0(38); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w27_1 <= DSP_bh5_ch1_0(37); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w26_1 <= DSP_bh5_ch1_0(36); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w25_1 <= DSP_bh5_ch1_0(35); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w24_1 <= DSP_bh5_ch1_0(34); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w23_1 <= DSP_bh5_ch1_0(33); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w22_1 <= DSP_bh5_ch1_0(32); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w21_1 <= DSP_bh5_ch1_0(31); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w20_1 <= DSP_bh5_ch1_0(30); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w19_1 <= DSP_bh5_ch1_0(29); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w18_1 <= DSP_bh5_ch1_0(28); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w17_1 <= DSP_bh5_ch1_0(27); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w16_1 <= DSP_bh5_ch1_0(26); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w15_1 <= DSP_bh5_ch1_0(25); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w14_1 <= DSP_bh5_ch1_0(24); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w13_1 <= DSP_bh5_ch1_0(23); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w12_1 <= DSP_bh5_ch1_0(22); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w11_1 <= DSP_bh5_ch1_0(21); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w10_1 <= DSP_bh5_ch1_0(20); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w9_1 <= DSP_bh5_ch1_0(19); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w8_1 <= DSP_bh5_ch1_0(18); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w7_1 <= DSP_bh5_ch1_0(17); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w6_0 <= DSP_bh5_ch1_0(16); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w5_0 <= DSP_bh5_ch1_0(15); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w4_0 <= DSP_bh5_ch1_0(14); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w3_0 <= DSP_bh5_ch1_0(13); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w2_0 <= DSP_bh5_ch1_0(12); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w1_0 <= DSP_bh5_ch1_0(11); -- cycle= 0 cp= 1.638e-09
   heap_bh5_w0_0 <= DSP_bh5_ch1_0(10); -- cycle= 0 cp= 1.638e-09
   ----------------Synchro barrier, entering cycle 0----------------

   -- Adding the constant bits

   ----------------Synchro barrier, entering cycle 0----------------
   ----------------Synchro barrier, entering cycle 0----------------
   ----------------Synchro barrier, entering cycle 1----------------
   finalAdderIn0_bh5 <= "0" & heap_bh5_w47_0_d1 & heap_bh5_w46_0_d1 & heap_bh5_w45_0_d1 & heap_bh5_w44_0_d1 & heap_bh5_w43_0_d1 & heap_bh5_w42_0_d1 & heap_bh5_w41_0_d1 & heap_bh5_w40_0_d1 & heap_bh5_w39_0_d1 & heap_bh5_w38_0_d1 & heap_bh5_w37_0_d1 & heap_bh5_w36_0_d1 & heap_bh5_w35_0_d1 & heap_bh5_w34_0_d1 & heap_bh5_w33_0_d1 & heap_bh5_w32_0_d1 & heap_bh5_w31_0_d1 & heap_bh5_w30_1_d1 & heap_bh5_w29_1_d1 & heap_bh5_w28_1_d1 & heap_bh5_w27_1_d1 & heap_bh5_w26_1_d1 & heap_bh5_w25_1_d1 & heap_bh5_w24_1_d1 & heap_bh5_w23_1_d1 & heap_bh5_w22_1_d1 & heap_bh5_w21_1_d1 & heap_bh5_w20_1_d1 & heap_bh5_w19_1_d1 & heap_bh5_w18_1_d1 & heap_bh5_w17_1_d1 & heap_bh5_w16_1_d1 & heap_bh5_w15_1_d1 & heap_bh5_w14_1_d1 & heap_bh5_w13_1_d1 & heap_bh5_w12_1_d1 & heap_bh5_w11_1_d1 & heap_bh5_w10_1_d1 & heap_bh5_w9_1_d1 & heap_bh5_w8_1_d1 & heap_bh5_w7_1_d1;
   finalAdderIn1_bh5 <= "0" & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & '0' & heap_bh5_w30_0_d1 & heap_bh5_w29_0_d1 & heap_bh5_w28_0_d1 & heap_bh5_w27_0_d1 & heap_bh5_w26_0_d1 & heap_bh5_w25_0_d1 & heap_bh5_w24_0_d1 & heap_bh5_w23_0_d1 & heap_bh5_w22_0_d1 & heap_bh5_w21_0_d1 & heap_bh5_w20_0_d1 & heap_bh5_w19_0_d1 & heap_bh5_w18_0_d1 & heap_bh5_w17_0_d1 & heap_bh5_w16_0_d1 & heap_bh5_w15_0_d1 & heap_bh5_w14_0_d1 & heap_bh5_w13_0_d1 & heap_bh5_w12_0_d1 & heap_bh5_w11_0_d1 & heap_bh5_w10_0_d1 & heap_bh5_w9_0_d1 & heap_bh5_w8_0_d1 & heap_bh5_w7_0_d1;
   finalAdderCin_bh5 <= '0';
   Adder_final5_0: IntAdder_42_f400_uid14  -- pipelineDepth=0 maxInDelay=0
      port map ( clk  => clk,
                 rst  => rst,
                 Cin => finalAdderCin_bh5,
                 R => finalAdderOut_bh5   ,
                 X => finalAdderIn0_bh5,
                 Y => finalAdderIn1_bh5);
   ----------------Synchro barrier, entering cycle 0----------------
   tempR_bh5_0 <= heap_bh5_w6_0 & heap_bh5_w5_0 & heap_bh5_w4_0 & heap_bh5_w3_0 & heap_bh5_w2_0 & heap_bh5_w1_0 & heap_bh5_w0_0; -- already compressed
   -- concatenate all the compressed chunks
   ----------------Synchro barrier, entering cycle 1----------------
   CompressionResult5 <= finalAdderOut_bh5 & tempR_bh5_0_d1;
   -- End of code generated by BitHeap::generateCompressorVHDL
   R <= CompressionResult5(47 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                       LeftShifter_48_by_max_4_uid23
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2011)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity LeftShifter_48_by_max_4_uid23 is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(47 downto 0);
          S : in  std_logic_vector(2 downto 0);
          R : out  std_logic_vector(51 downto 0)   );
end entity;

architecture arch of LeftShifter_48_by_max_4_uid23 is
signal level0, level0_d1 :  std_logic_vector(47 downto 0);
signal ps, ps_d1 :  std_logic_vector(2 downto 0);
signal level1 :  std_logic_vector(48 downto 0);
signal level2 :  std_logic_vector(50 downto 0);
signal level3 :  std_logic_vector(54 downto 0);
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            level0_d1 <=  level0;
            ps_d1 <=  ps;
         end if;
      end process;
   level0<= X;
   ps<= S;
   ----------------Synchro barrier, entering cycle 1----------------
   level1<= level0_d1 & (0 downto 0 => '0') when ps_d1(0)= '1' else     (0 downto 0 => '0') & level0_d1;
   level2<= level1 & (1 downto 0 => '0') when ps_d1(1)= '1' else     (1 downto 0 => '0') & level1;
   level3<= level2 & (3 downto 0 => '0') when ps_d1(2)= '1' else     (3 downto 0 => '0') & level2;
   R <= level3(51 downto 0);
end architecture;

--------------------------------------------------------------------------------
--                            LongAcc_5_23_3_M1_25
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin, Bogdan Pasca (2008-2009)
--------------------------------------------------------------------------------
-- Pipeline depth: 2 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity LongAcc_5_23_3_M1_25 is
   port ( clk, rst : in std_logic;
          sigX_dprod : in std_logic;
          excX_dprod : in  std_logic_vector(1 downto 0);
          fracX_dprod : in  std_logic_vector(47 downto 0);
          expX_dprod : in  std_logic_vector(4 downto 0);
          newDataSet : in std_logic;
          ready : out std_logic;
          A : out  std_logic_vector(26 downto 0);
          C : out  std_logic_vector(26 downto 0);
          XOverflow : out std_logic;
          XUnderflow : out std_logic;
          AccOverflow : out std_logic   );
end entity;

architecture arch of LongAcc_5_23_3_M1_25 is
   component LeftShifter_48_by_max_4_uid23 is
      port ( clk, rst : in std_logic;
             X : in  std_logic_vector(47 downto 0);
             S : in  std_logic_vector(2 downto 0);
             R : out  std_logic_vector(51 downto 0)   );
   end component;

signal fracX :  std_logic_vector(47 downto 0);
signal expX :  std_logic_vector(4 downto 0);
signal signX, signX_d1 : std_logic;
signal exnX, exnX_d1 :  std_logic_vector(1 downto 0);
signal xOverflowCond, xOverflowCond_d1, xOverflowCond_d2 : std_logic;
signal xUnderflowCond, xUnderflowCond_d1, xUnderflowCond_d2 : std_logic;
signal shiftVal, shiftVal_d1 :  std_logic_vector(5 downto 0);
signal shifted_frac :  std_logic_vector(51 downto 0);
signal flushedToZero : std_logic;
signal summand :  std_logic_vector(4 downto 0);
signal summand2c :  std_logic_vector(4 downto 0);
signal ext_summand2c, ext_summand2c_d1 :  std_logic_vector(26 downto 0);
signal carryBit_0, carryBit_0_d1 : std_logic;
signal acc_0, acc_0_d1 :  std_logic_vector(26 downto 0);
signal carryBit_1, carryBit_1_d1 : std_logic;
signal acc_0_ext :  std_logic_vector(27 downto 0);
signal xOverflowRegister, xOverflowRegister_d1 : std_logic;
signal xUnderflowRegister, xUnderflowRegister_d1 : std_logic;
signal accOverflowRegister, accOverflowRegister_d1 : std_logic;
signal acc :  std_logic_vector(26 downto 0);
signal carry :  std_logic_vector(26 downto 0);
signal newDataSet_d1, newDataSet_d2 : std_logic;
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            signX_d1 <=  signX;
            exnX_d1 <=  exnX;
            shiftVal_d1 <=  shiftVal;
            ext_summand2c_d1 <=  ext_summand2c;
            carryBit_0_d1 <=  carryBit_0;
            newDataSet_d1 <=  newDataSet;
            newDataSet_d2 <=  newDataSet_d1;
         end if;
      end process;
   process(clk, rst)
      begin
         if clk'event and clk = '1' then
            if rst = '1' then
               xOverflowCond_d1 <=  '0';
               xOverflowCond_d2 <=  '0';
               xUnderflowCond_d1 <=  '0';
               xUnderflowCond_d2 <=  '0';
               acc_0_d1 <=  (others => '0');
               carryBit_1_d1 <=  '0';
               xOverflowRegister_d1 <=  '0';
               xUnderflowRegister_d1 <=  '0';
               accOverflowRegister_d1 <=  '0';
            else
               xOverflowCond_d1 <=  xOverflowCond;
               xOverflowCond_d2 <=  xOverflowCond_d1;
               xUnderflowCond_d1 <=  xUnderflowCond;
               xUnderflowCond_d2 <=  xUnderflowCond_d1;
               acc_0_d1 <=  acc_0;
               carryBit_1_d1 <=  carryBit_1;
               xOverflowRegister_d1 <=  xOverflowRegister;
               xUnderflowRegister_d1 <=  xUnderflowRegister;
               accOverflowRegister_d1 <=  accOverflowRegister;
            end if;
         end if;
      end process;
   fracX <= fracX_dprod;
   expX <= expX_dprod;
   signX <= sigX_dprod;
   exnX <= excX_dprod;
   xOverflowCond <= '1' when (( expX > CONV_STD_LOGIC_VECTOR(18,5)) or (exnX >= "10")) else '0' ;
   xUnderflowCond <= '1' when (expX < CONV_STD_LOGIC_VECTOR(14,5)) else '0' ;
   shiftVal <= ("0" & expX) - CONV_STD_LOGIC_VECTOR(14,6);
   LongAccInputShifter: LeftShifter_48_by_max_4_uid23  -- pipelineDepth=1 maxInDelay=1.8946e-09
      port map ( clk  => clk,
                 rst  => rst,
                 R => shifted_frac,
                 S => shiftVal(2 downto 0),
                 X => fracX);
   ----------------Synchro barrier, entering cycle 1----------------
   flushedToZero <= '1' when (shiftVal_d1(5)='1' or exnX_d1="00") else '0';
   summand<= "00000" when flushedToZero='1' else shifted_frac(27 downto 23);
   -- 2's complement of the summand
   summand2c <= summand when (signX_d1='0' or flushedToZero='1') else not(summand);
   -- sign extension of the summand to accumulator size
   ext_summand2c <= (26 downto 5 => signX_d1 and not flushedToZero) & summand2c;
   -- accumulation itself
   carryBit_0 <= signX_d1 and not flushedToZero;
   acc_0 <= acc_0_ext(26 downto 0);
   carryBit_1  <= acc_0_ext(27);
   ----------------Synchro barrier, entering cycle 2----------------
   acc_0_ext <= ( "0" & (acc_0_d1 and (26 downto 0 => not(newDataSet_d2)))) + ( "0" & ext_summand2c_d1(26 downto 0)) + (carryBit_0_d1);
   ---------------- cycle 1----------------
   xOverflowRegister <= xOverflowRegister_d1 or xOverflowCond_d2;
   xUnderflowRegister <= xUnderflowRegister_d1 or xUnderflowCond_d2;
   accOverflowRegister <= accOverflowRegister_d1 or carryBit_1_d1;
   ----------------Synchro barrier, entering cycle 2----------------
   acc <= acc_0_d1;
   carry <= "000000000000000000000000000";
   A <=  acc_0_d1;
   C <=   carry;
   AccOverflow <= accOverflowRegister_d1;
   XOverflow <= xOverflowRegister_d1;
   XUnderflow <= xUnderflowRegister_d1;
   ---------------- cycle 2----------------
   ready <= newDataSet_d2;
end architecture;

--------------------------------------------------------------------------------
--                       DotProduct_5_23_23_3_M1_25400
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2011)
--------------------------------------------------------------------------------
-- Pipeline depth: 1 cycles

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity DotProduct_5_23_23_3_M1_25400 is
   port ( clk, rst : in std_logic;
          X : in  std_logic_vector(5+23+2 downto 0);
          Y : in  std_logic_vector(5+23+2 downto 0);
          newDataSet : in std_logic;
          A : out  std_logic_vector(26 downto 0);
          C : out  std_logic_vector(26 downto 0);
          XOverflow : out std_logic;
          XUnderflow : out std_logic;
          AccOverflow : out std_logic   );
end entity;

architecture arch of DotProduct_5_23_23_3_M1_25400 is
   component IntMultiplier_UsingDSP_24_24_0_unsigned_uid3 is
      port ( clk, rst : in std_logic;
             X : in  std_logic_vector(23 downto 0);
             Y : in  std_logic_vector(23 downto 0);
             R : out  std_logic_vector(47 downto 0)   );
   end component;

   component LongAcc_5_23_3_M1_25 is
      port ( clk, rst : in std_logic;
             sigX_dprod : in std_logic;
             excX_dprod : in  std_logic_vector(1 downto 0);
             fracX_dprod : in  std_logic_vector(47 downto 0);
             expX_dprod : in  std_logic_vector(4 downto 0);
             newDataSet : in std_logic;
             ready : out std_logic;
             A : out  std_logic_vector(26 downto 0);
             C : out  std_logic_vector(26 downto 0);
             XOverflow : out std_logic;
             XUnderflow : out std_logic;
             AccOverflow : out std_logic   );
   end component;

signal sX : std_logic;
signal sY : std_logic;
signal excX :  std_logic_vector(1 downto 0);
signal excY :  std_logic_vector(1 downto 0);
signal expX :  std_logic_vector(4 downto 0);
signal expY :  std_logic_vector(4 downto 0);
signal fracX :  std_logic_vector(23 downto 0);
signal fracY :  std_logic_vector(23 downto 0);
signal signP, signP_d1 : std_logic;
signal mFrac :  std_logic_vector(47 downto 0);
signal sumExp :  std_logic_vector(5 downto 0);
signal overflow : std_logic;
signal sumExpMBias :  std_logic_vector(5 downto 0);
signal underflow : std_logic;
signal excConcat :  std_logic_vector(3 downto 0);
signal excPhase1, excPhase1_d1 :  std_logic_vector(1 downto 0);
signal excConcatOU, excConcatOU_d1 :  std_logic_vector(3 downto 0);
signal exc :  std_logic_vector(1 downto 0);
signal accReady : std_logic;
signal accA :  std_logic_vector(26 downto 0);
signal accC :  std_logic_vector(26 downto 0);
signal accXOverflow : std_logic;
signal accXUnderflow : std_logic;
signal accAccOverflow : std_logic;
signal newDataSet_d1 : std_logic;
begin
   process(clk)
      begin
         if clk'event and clk = '1' then
            signP_d1 <=  signP;
            excPhase1_d1 <=  excPhase1;
            excConcatOU_d1 <=  excConcatOU;
            newDataSet_d1 <=  newDataSet;
         end if;
      end process;
   sX <= X(30);
   sY <= Y(30);
   excX <= X(29 downto 28);
   excY <= Y(29 downto 28);
   expX <= X(27 downto 23);
   expY <= Y(27 downto 23);
   fracX <= "1" & X(22 downto 0);
   fracY <= "1" & Y(22 downto 0);
   signP <= sX xor sY;
   MantisaMultiplier: IntMultiplier_UsingDSP_24_24_0_unsigned_uid3  -- pipelineDepth=1 maxInDelay=3.2865e-10
      port map ( clk  => clk,
                 rst  => rst,
                 R => mFrac   ,
                 X => fracX,
                 Y => fracY);
   sumExp <= ("0" & expX) + ("0" & expY);
   overflow <= sumExp(5);
   sumExpMBias <= sumExp - ("0" & CONV_STD_LOGIC_VECTOR(15,5));
   underflow <= sumExpMBias(5);
   excConcat<= excX & excY;
    with excConcat select 
   excPhase1 <=  "00" when "0000",
      "01" when "0101",
      "10" when "1001"|"0110"|"1010",
      "11" when others;
   excConcatOU <= excPhase1 & overflow & underflow;
   ----------------Synchro barrier, entering cycle 1----------------
    with excConcatOU_d1 select 
   exc <=  "00" when "0101",
      "10" when "0110",
      excPhase1_d1 when others;
   Accumulator: LongAcc_5_23_3_M1_25  -- pipelineDepth=2 maxInDelay=0
      port map ( clk  => clk,
                 rst  => rst,
                 A => accA,
                 AccOverflow => accAccOverflow,
                 C => accC,
                 XOverflow => accXOverflow,
                 XUnderflow => accXUnderflow,
                 excX_dprod => exc,
                 expX_dprod => sumExpMBias(4 downto 0),
                 fracX_dprod => mFrac,
                 newDataSet => newDataSet_d1,
                 ready => accReady   ,
                 sigX_dprod => signP_d1);
   A <= accA;
   C <= accC;
   XOverflow <= accXOverflow;
   XUnderflow <= accXUnderflow;
   AccOverflow <= accAccOverflow;
end architecture;


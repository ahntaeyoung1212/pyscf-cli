# 講義「マテリアル計算科学」× pyscf-cli 対応マップ

pyscf-cli は東京理科大学「マテリアル計算科学」(全15回、望月泰英)の演習
スクリプト群から生まれました。このドキュメントは、講義の各回と pyscf-cli の
機能・演習の対応表です。同様の量子化学・計算材料科学の講義を設計する際の
テンプレートとしてもお使いください。

## コースの5つのアーク

| アーク | 回 | 内容 |
|---|---|---|
| A. 導入 | 第1回 | 「構造→物性」の思想。計算の3つの型(全エネルギー/電子準位/振動) |
| B. 解析的量子力学 | 第2〜4回 | 厳密解→変分原理→摂動論。紙と鉛筆で He を追い詰める |
| C. 基底関数とCLI入門 | 第5〜6回 | 重なり積分の手計算→STO-3G→**PySCF 解禁** |
| D. 多電子系の実践 | 第7〜10回 | スピン・BO近似・交換エネルギー→**第10回で全機能総動員** |
| E. 結合と固体 | 第11〜14回 | 永年方程式→群論→Madelung→イオン結晶(後半は固体・範囲外) |

「3つの型」は pyscf-cli のサブコマンド体系にそのまま対応します:
型1=`energy`/`relax`/`thermo`、型2=`orbitals`/`dos`、型3=`vib`/`vibmovie`。

## 講義対応表

| 回 | 主題 | 対応する pyscf-cli 演習 |
|---|---|---|
| 1 | 計算科学の3つの型、虚振動と構造不安定性 | 全サブコマンドの見取り図([TEACHING_ja](TEACHING_ja.md) 序文) |
| 2 | H原子の厳密解、量子数、⟨r⟩ | `orbitals h.xyz --spin 1`(1s の実物)、`info basis` |
| 3 | 変分原理、He の電子間反発(27%誤差)、Z'=27/16 | `energy he.xyz --decompose-total-energy`(演習9) |
| 4 | 摂動論、He 1次摂動、相関エネルギーの姿 | `energy he.xyz --theory ccsd`(演習10 数値ラダー) |
| 5 | 重なり積分の手計算、STO-3G の導出、**PySCF 初登場** | `energy h.xyz --spin 1 --basis sto-3g / 6-31g`(tutorial1) |
| 6 | STO-1G/3G の誤差、RHF/UHF/ROHF | 基底比較(演習7)、`--method` 比較(演習11') |
| 7 | スピン、Slater行列式、項記号、ΔSCF | Be 励起状態コース(演習3)、C のフント則(演習11) |
| 8 | Born-Oppenheimer、Hellmann-Feynman、O₂ PES | `energy --pes`(演習1)、O₂ 解離エネルギー(演習5') |
| 9 | クーロン積分 U と交換積分 J、フント則の起源 | `--decompose-total-energy`(演習9)、`--fixed-occ-decomp` |
| 10 | **Molcalc 全機能の PySCF 再現**(実習の頂点) | `convert`→`relax`→`energy`→`dos`→`orbitals`→`vib` 一気通貫(演習14) |
| 11 | 永年方程式、共有結合、α・β・S | `orbitals h2.xyz`(σg/σu*)、`--occ-alpha` で反結合占有 |
| 12 | 群論、対称性と共有結合、O₂ ³Σg⁻ | `dos ch4.xyz`(t₂:a₁ 縮退)、`vib h2o.xyz`(A₁/B₁ モード) |
| 13 | vdW力の導出、Madelung エネルギー | 範囲外(pymatgen_tutorial + calc_madelung_constant.py を使用)。接点: ZPE と vdW の物理 |
| 14 | イオン結晶の電子状態、バンドギャップ | 分子側からの橋: NaCl イオン化収支(演習12)、`dos` の HOMO-LUMO ギャップ |

## チュートリアル zip との対応

| 配布物 | 対応講義 | pyscf-cli での等価物 |
|---|---|---|
| pyscf_tutorial1 | 第5〜6回 | `examples` の単原子 XYZ + `energy` 基底比較 |
| pyscf_tutorial2 | 第6〜7回 | `--method rhf/uhf/rohf` 比較、`dos`、`relax` |
| pyscf_tutorial3 | 第8回 | `energy --pes`(H₂/O₂、bash ループ不要に) |
| pyscf_tutorial4 | 第10回〜 | `convert`(SDF→XYZ)+ `dos --coop --cohp` + 全パイプライン |
| pymatgen_tutorial | 第13〜14回 | **対象外**(固体・周期系)。Colab + calc_madelung_constant.py を継続使用 |

## スコープの整理

pyscf-cli は**分子の量子化学**に特化します。第13〜14回の固体パート
(POSCAR、Ewald和、バンド構造)は pymatgen ベースの教材を引き続き使用して
ください。ただし「分子から固体への橋」になる演習(NaCl のイオン化収支、
HOMO-LUMO ギャップ=バンドギャップの分子版)は演習集に収録しています。

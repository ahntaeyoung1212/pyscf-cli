# pyscf-cli

**XYZファイル1枚とコマンド1行で、本物の量子化学計算を。**

`pyscf-cli` は [PySCF](https://pyscf.org/) の教育用フロントエンドです。
Python を書かずに、Hartree–Fock・DFT・MP2・CCSD(T) の一点計算、構造最適化、
振動解析、熱化学、分子DOS図、軌道可視化、Δ-SCF 励起状態までを実行できます。
中身を知りたくなったら `--dry-run` — どの計算についても等価な PySCF
Python スクリプトを表示します。

> **注意:** pyscf-cli は東京理科大学 望月研究室が開発する独立の教育プロジェクト
> であり、PySCF 開発チームの公式ツールでは**ありません**。学術利用の際は
> PySCF 本体を引用してください([引用](#引用)参照)。

## インストール

```bash
pip install pyscf-cli        # (PyPI 公開後)
# 開発版:
pip install git+https://github.com/ahntaeyoung1212/pyscf-cli.git
```

Python 3.9 以上、**Linux または macOS** で動作します。PySCF は Windows を
ネイティブサポートしないため、Windows の学生には WSL か **Google Colab**
(`%pip install pyscf-cli` で動きます)を推奨してください。

## 5分クイックスタート

```bash
pyscf-cli examples h2o          # サンプル分子を ./h2o.xyz にコピー
pyscf-cli energy h2o.xyz        # RHF/STO-3G 一点計算
pyscf-cli examples              # 同梱分子の一覧
pyscf-cli info                  # 基底関数系・汎関数の早見表
```

## できること

| コマンド | 内容 |
|---|---|
| `energy` | 全エネルギー(HF/DFT/MP2/CCSD/CCSD(T))、MO準位、⟨S²⟩、二原子分子PESスキャン、エネルギー分解、Δ-SCF励起状態 |
| `relax` | 構造最適化(geomeTRIC)。`<入力名>-finish.xyz` を出力 |
| `vib` | 調和振動数、虚振動の検出、振動準位 E_n |
| `thermo` | ZPE、E/H/G/S/Cp(温度・圧力指定可) |
| `dos` | 分子DOS/PDOS図(s/p/d/f・元素分解、Löwdin/Mulliken、スピン分解)+ COOP/COHP 結合性解析 |
| `orbitals` | MO の cube ファイル出力(VESTA/Avogadro 用) |
| `vibmovie` | 振動モードごとの GIF アニメーション |
| `convert` | SDF → XYZ 変換(PubChem ダウンロード→計算のパイプライン) |
| `examples` | サンプル18種: 分子(H₂O, O₂, CO₂, NH₃, CH₄, ベンゼン等)+ 単原子(H〜Ne, Na, Cl) |
| `info` | 基底関数系・汎関数・理論レベルの解説付き一覧 |

使用例:

```bash
pyscf-cli energy o2.xyz --spin 2 --method uhf              # 三重項 O2(⟨S²⟩=2 を確認!)
pyscf-cli energy h2o.xyz --theory dft --xc b3lyp --basis 6-31g**
pyscf-cli energy h2.xyz --pes --rmin 0.4 --rmax 3.0        # 結合解離曲線
pyscf-cli relax h2o.xyz --basis 6-31g
pyscf-cli vib h2o.xyz --basis 6-31g
pyscf-cli thermo h2o.xyz --basis 6-31g --temp 298.15
pyscf-cli dos benzene.xyz --element-pdos --align homo
pyscf-cli orbitals h2o.xyz --homo --lumo                   # VESTA 用 cube
pyscf-cli vibmovie h2o.xyz --basis 6-31g
```

## 教育のための設計

- **エラーが教材** — 基底名の typo には候補を提示、不可能な電荷/スピンの
  組には「--spin は不対電子数」と解説、鞍点ジオメトリには「極小ではない」
  警告。
- **`--dry-run` は PySCF への架け橋** — CLI と等価な実行可能 Python
  スクリプトを表示。CLI → スクリプトを読む → 編集する、の順で卒業できます。
- **`--json` で自動採点** — 全計算コマンドが機械可読出力に対応
  (`--json result.json`、`--json -` で純JSON)。終了コードは
  0=成功 / 2=入力エラー / 3=SCF未収束。
- **電子配置の制御** — `--spin` で不対電子数、`--occ-alpha 1,3` で任意の
  MO 占有を指定し、その配置を保持したまま SCF(最大重なり法)。収束後の
  ⟨S²⟩ が「本当にその状態か」の検算になります。
- **エネルギー分解** — `--decompose-total-energy` で E = T + V_ne + U + J
  + V_nn(総和が E_SCF に一致することをテストで保証)。
  `--fixed-occ-decomp` は凍結軌道昇位を解析し、交換積分 K と
  一重項/三重項推定値まで表示します。

授業でそのまま使える演習例は [docs/TEACHING_ja.md](docs/TEACHING_ja.md) へ。

## 引用

pyscf-cli を授業・研究で使う場合は、実際の量子化学計算を担う **PySCF** を
引用してください:

> Q. Sun *et al.*, "Recent developments in the PySCF program package",
> *J. Chem. Phys.* **153**, 024109 (2020). DOI: 10.1063/5.0006074

pyscf-cli 自体の DOI(Zenodo)は正式リリース時に発行予定です。

## ライセンス

MIT © 2026 望月泰英(東京理科大学)。PySCF 本体は Apache-2.0 です。

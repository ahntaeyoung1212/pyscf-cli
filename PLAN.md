# PLAN.md — `pyscf-cli` 開発計画

**XYZファイル1枚とコマンド1行で量子化学計算を実行できる、教育向け PySCF フロントエンド**

作成日: 2026-07-17 / 作成者: Mochizuki group (Tokyo Univ of Science) + Claude

---

## 0. 事前調査の結果(なぜ作る価値があるか)

| 調査対象 | 結果 |
|---|---|
| PyPI `pyscf-cli` | **未登録**(名前は取得可能) |
| PyPI `pyscfcli` / `pyscf-tools` / `qc-cli` | すべて未登録 |
| GitHub [pyscf/pyscfcli](https://github.com/pyscf/pyscfcli) | 存在するが**別コンセプト**。YAML/JSON/TOML 設定ファイルから Python スクリプトを生成する「コンパイラ」型。8 コミット・リリースなし・PyPI 未公開・事実上停止 |
| 教育向け「XYZ+フラグ→即計算」型 CLI | 見当たらない |

**結論:** 「学生が XYZ ファイルとコマンドライン引数だけで SCF/DFT/MP2/CCSD(T)・構造最適化・振動・熱化学・DOS・軌道可視化まで実行できる」ツールはニッチが空いている。既存の選択肢は (a) Gaussian/ORCA = 有償 or 登録制でブラックボックス、(b) 生 PySCF = Python API の学習コストが高い。その中間を埋める。

### 命名に関する注意(要認識)

- `pyscf-cli` という名前は PyPI 規約上問題なく取得できる(`pyscf-dispersion` 等、pyscf- プレフィックスの前例あり)。
- ただし PySCF 公式 org が GitHub に `pyscfcli` を持つため、**README 冒頭に「本パッケージは PySCF 開発チームの公式ツールではない」旨を明記**して混同を避ける(Apache-2.0 の商標条項的にも誠実)。
- 公開が軌道に乗ったら PySCF チームに一報を入れて "extension ecosystem" ページへの掲載を打診する価値あり(公認への近道)。

---

## 1. 目的とターゲットユーザー

| ユーザー | ニーズ |
|---|---|
| 学部生・院生(授業の受講者) | プログラミング知識ゼロで計算を回し、結果(数値・図・アニメ)を得る |
| 授業担当教員(自分+他大学) | 演習教材としてそのまま採用できる。インストール指示が1行で済む |
| 研究室の新人 | PySCF 本体に進む前の入門ステップ |

**設計原則(すべての判断基準):**
1. **最初の5分で成功体験** — `pip install pyscf-cli` → サンプル取得 → 計算成功、までを5分以内に。
2. **エラーは教材** — 失敗時のメッセージは「何が悪いか+どう直すか」を学生向けの言葉で。
3. **出力は講義資料にそのまま使える品質**(整形テキスト、PDF/PNG 図、CSV)。
4. **PySCF への卒業を促す** — `--dry-run` で等価な PySCF Python スクリプトを表示し、CLI の背後にあるコードを見せる(pyscf/pyscfcli の良いアイデアを拝借)。

---

## 2. CLI 設計

### 2.1 コマンド体系: サブコマンド型(決定済み)

```
pyscf-cli <subcommand> [options]
```

| サブコマンド | 由来スクリプト | 機能 |
|---|---|---|
| `energy` | calc_pyscf.py | 一点計算(SCF/DFT/MP2/CCSD/CCSD(T))、MO、⟨S²⟩、PES、MOM、エネルギー分解 |
| `relax` | calc_pyscf_relax.py | 構造最適化(geomeTRIC) |
| `vib` | calc_pyscf_vib.py | 振動数解析+調和振動準位 |
| `thermo` | calc_pyscf_thermo.py | 熱化学(ZPE/H/G/S/Cp) |
| `dos` | calc_pyscf_dos.py | 分子 DOS/PDOS プロット |
| `orbitals` | calc_pyscf_wf.py | MO cube 出力(VESTA 用)※ `wf` より意味が明確な名前に変更 |
| `vibmovie` | calc_pyscf_vib_movie.py | 振動モード GIF アニメ |
| `examples` | (新規) | サンプル XYZ をカレントにコピー(`pyscf-cli examples h2o` 等) |
| `info` | (新規) | 利用可能な基底・汎関数の一覧表示、バージョン情報 |

移行例:

```bash
# 旧
python3 calc_pyscf.py --xyz XYZ_O2.xyz --spin 0 --basis 6-31g --charge 0 --theory scf --method uhf
# 新
pyscf-cli energy --xyz XYZ_O2.xyz --spin 0 --basis 6-31g --method uhf

# 旧
python3 calc_pyscf_relax.py --xyz XYZ_H2O.xyz --spin 0 --basis "6-31g**"
# 新
pyscf-cli relax --xyz XYZ_H2O.xyz --basis "6-31g**"
```

### 2.2 共通オプション(全サブコマンドで統一)

現行スクリプトの良い点(統一された `--xyz --basis --theory --xc --spin --charge --unit`)をそのまま共通基盤化する。

```
--xyz FILE          入力XYZ(位置引数としても受理: pyscf-cli energy h2o.xyz)
--basis BASIS       基底関数系 (default: sto-3g / 精度系は 6-31g)
--theory {scf,dft,mp2,ccsd,ccsd_t}
--xc FUNCTIONAL     DFT汎関数 (default: b3lyp)
--method {auto,rhf,uhf,rohf}   (default: auto = spin から自動判定)
--spin 2S / --charge Q / --unit {Angstrom,Bohr}
--json              結果を機械可読 JSON でも出力(採点スクリプト・自動化用)
--dry-run           実行せず、等価な PySCF Python スクリプトを表示(教育機能)
-q / --quiet, -v / --verbose
```

### 2.3 ユーザビリティ向上策(熟慮ポイント)

1. **位置引数で XYZ を受理** — `pyscf-cli energy h2o.xyz` を許可(`--xyz` も後方互換で残す)。タイプ量最小化。
2. **賢いエラーメッセージ** — 例:
   - 基底名 typo → `Unknown basis '6-31g*8'. Did you mean '6-31g**'?`(難読な PySCF トレースバックを見せない)
   - O₂ を `--spin 0` で計算 → 収束後に ⟨S²⟩ 異常や不安定性を検知したら「O₂ は三重項が基底状態です。`--spin 2` を試してください」とヒントを出す(教材化)
   - SCF 不収束 → 「初期値・基底・スピンを疑え」の定型ガイダンス
3. **`pyscf-cli examples`** — `h2.xyz` `h2o.xyz` `o2.xyz` `co2.xyz` `benzene.xyz` 等をパッケージ同梱し、1コマンドでカレントへコピー。「教員がファイル配布する手間」を消す。
4. **`pyscf-cli info basis` / `info xc`** — choices の一覧+一言説明(「6-31g**: 分極関数付き。有機分子の標準的入門用」)。argparse の choices エラーより遥かに親切。
5. **単位の並記** — 全エネルギー出力で Hartree と eV を常に併記(現行は箇所によって片方のみ)。教育上、単位換算の意識付けに有効。
6. **計算時間の目安警告** — 原子数×基底サイズから「CCSD(T)/cc-pvqz は数時間かかる可能性」等を実行前に警告(授業中のフリーズ体験を防ぐ)。
7. **`--dry-run`(スクリプト表示)** — 「CLI で慣れる→生成スクリプトを読む→PySCF を直接書く」の学習曲線を作る。差別化機能。
8. **終了コードの規約** — 正常 0 / 入力エラー 2 / 収束失敗 3 等。教員の自動採点・CI で使える。
9. **出力ファイル名の一貫性** — `<input名>_energy.txt`, `<input名>_dos.pdf` 等、入力名ベースで統一(現行の `DOS_`, `relax_`, `WF_` プレフィックスの混在を整理)。
10. **Google Colab 対応を第一級市民に** — PySCF は Windows ネイティブ非対応(WSL 必須)。教室の現実解は Colab。`!pip install pyscf-cli` で動く Colab ノートブックを examples として同梱・リンク。**これが「他大学でも使える」の実質的な鍵。**

---

## 3. パッケージ構成

```
pyscf-cli/                       # GitHub リポジトリ名
├── LICENSE                      # MIT
├── README.md                    # 英語(国際標準)。冒頭に非公式 disclaimer
├── README_ja.md                 # 日本語版
├── CHANGELOG.md
├── pyproject.toml               # PEP 621。[project.scripts] pyscf-cli = "pyscf_cli.main:main"
├── src/
│   └── pyscf_cli/
│       ├── __init__.py          # __version__
│       ├── main.py              # argparse subparsers → 各モジュールへ dispatch
│       ├── core.py              # ★重複解消: read_xyz, write_xyz, build_mf, build_ks,
│       │                        #   run_theory, 基底/汎関数リスト, 定数, 共通引数定義
│       ├── output.py            # 整形出力・JSON 出力・単位換算の一元化
│       ├── energy.py            # 旧 calc_pyscf.py
│       ├── relax.py / vib.py / thermo.py / dos.py / orbitals.py / vibmovie.py
│       └── data/                # サンプル XYZ (importlib.resources で読む)
├── tests/
│   ├── test_core.py             # XYZ パーサ、引数検証
│   ├── test_energy.py           # H2/H2O の RHF/STO-3G 等、既知値との回帰テスト
│   └── test_smoke.py            # 全サブコマンドが --help と最小実行で死なないこと
├── examples/
│   └── colab_quickstart.ipynb   # Colab バッジ付きノートブック
└── .github/workflows/
    ├── ci.yml                   # pytest (Linux + macOS, Python 3.10–3.13)
    └── publish.yml              # タグ push → PyPI Trusted Publishing
```

### 依存関係

```toml
[project]
requires-python = ">=3.10"
dependencies = [
  "pyscf>=2.3",
  "numpy",
  "matplotlib",       # dos, vibmovie
  "geometric",        # relax
]
[project.optional-dependencies]
vesta-colors = ["pymatgen"]   # dos の VESTA 配色(fallback 内蔵済みなので任意)
dev = ["pytest", "ruff"]
```

方針: **教育用途ゆえ「全部入りデフォルト」**。extras 分割で `relax` が動かない事故を防ぐ。重い pymatgen のみ任意(現行コードに fallback 配色が既にある)。

---

## 4. 実装フェーズ

### Phase 1: 基盤 (リファクタリング)
- [ ] `git init`、リポジトリ雛形、pyproject.toml、MIT LICENSE
- [ ] `core.py` 作成 — 7 スクリプトに重複する `read_xyz` / `build_mf` / 基底リスト等を集約
- [ ] `output.py` — Hartree/eV 併記、JSON 出力、定型ヘッダの一元化
- [ ] `main.py` — サブコマンドディスパッチ + 共通引数の親パーサ

### Phase 2: 移植
- [ ] 7 スクリプトを各モジュールへ移植(ロジックは極力温存、入出力周りのみ core/output 経由に)
- [ ] `examples` / `info` サブコマンド新規実装
- [ ] 位置引数 XYZ、エラーメッセージ改善、`--dry-run`
- [ ] 既存機能の動作同等性を旧スクリプトと突き合わせ確認

### Phase 3: 品質保証
- [x] 回帰テスト: H₂/H₂O/O₂ の RHF・UHF・B3LYP エネルギー既知値(許容誤差 1e-6 Ha)
- [x] エネルギー分解(T+V_ne+J+K+V_nn = E_tot)の恒等式テスト
- [x] 振動数: H₂O の 3 モードが文献値 ±数% に入ること
- [x] スモークテスト: 全サブコマンド × 最小分子
- [ ] GitHub Actions CI(Linux/macOS × Python 3.10–3.13)
- [x] 公開前コードレビュー(特に UHF 交換項・spin_square・MOM まわりの数式検証)

### Phase 4: ドキュメント
- [ ] README.md(英)/ README_ja.md — インストール、5 分クイックスタート、全サブコマンド例と出力サンプル、Colab バッジ
- [ ] 非公式 disclaimer + PySCF 引用のお願い(PySCF の WIREs/JCP 論文)
- [ ] Colab ノートブック(授業でそのまま配れる体裁)
- [ ] 教員向け: 「演習アイデア集」(PES で結合解離、O₂ のスピン状態、DOS で共役系、等)

### Phase 5: 公開
- [ ] **TestPyPI に先行リリース** → クリーンな venv と Colab で install 検証
- [ ] PyPI 本公開(Trusted Publishing、API トークン不要の GitHub OIDC 方式)
- [ ] GitHub リポジトリ公開、v1.0.0 タグ、Zenodo DOI 取得(授業シラバス・論文から引用可能に)
- [ ] PySCF チームへ連絡(ecosystem 掲載打診)

### Phase 6: 公開後(任意)
- [ ] 他大学教員へのアナウンス(学会 ML、分子科学討論会等)
- [ ] **JOSS (Journal of Open Source Software) 投稿を検討** — 教育用 OSS は対象内。査読付き論文として業績化できる
- [ ] conda-forge 登録(PySCF 本体が conda 利用者に多いため)
- [ ] 要望に応じて: TDDFT/UV-Vis、NMR、溶媒効果(PCM)等のサブコマンド追加

---

## 5. ライセンス整理(確認済み事項)

- **PySCF は Apache-2.0** → import して呼ぶだけの本パッケージは任意のライセンスを選べる。**MIT で公開(決定)**。
- PySCF のソースを同梱・改変しない限り、Apache-2.0 の義務(LICENSE/NOTICE 同梱等)は発生しない。
- README に記載すること:
  1. 非公式である旨(PySCF 商標への配慮)
  2. 「学術利用時は PySCF の論文を引用してください」
  3. 本パッケージ自体の引用方法(Zenodo DOI)

---

## 6. リスクと対策

| リスク | 対策 |
|---|---|
| PySCF 公式が将来 `pyscf-cli` 名で公式ツールを出す | 早期に公開して既成事実化 + 公式チームに連絡して関係構築。最悪時はリネーム(`qcedu` 等の代替案は温存) |
| Windows 学生がローカルで動かせない | README に「Windows は WSL または Colab」を明記。Colab を第一の推奨経路に |
| CCSD(T)/大基底で授業マシンがフリーズ | 実行前のコスト警告(2.3-6)+ ドキュメントで推奨レベルを明示 |
| メンテ負担(研究・授業と並行) | v1.0 のスコープを現行 7 機能+2 新規に固定。Issue テンプレートで「教育用途優先」を明示 |
| 数値の誤り(教材として致命的) | Phase 3 の恒等式テスト+既知値回帰を必須ゲートに |

---

## 7. 当面のマイルストーン

| 時期 | 目標 |
|---|---|
| Week 1 | Phase 1–2 完了(リファクタ+移植、ローカルで `pipx install -e .` 動作) |
| Week 2 | Phase 3(テスト+CI 緑)+ Phase 4 ドラフト |
| Week 3 | TestPyPI → 検証 → **PyPI v1.0.0 公開** |
| 後期授業前 | Colab 教材整備、授業で実戦投入 → フィードバック反映 v1.1 |

# pyscf-cli 演習アイデア集(教員向け)

量子化学の講義・学生実験で pyscf-cli をそのまま使うための演習例です。
各演習に、コマンド、学生への問いかけ、所要時間の目安を付けています。
計算はすべてノートPC(または Google Colab)で数秒〜数分で終わるよう、
小さな分子と基底で設計しています。

東京理科大学「マテリアル計算科学」(全15回)との講義対応は
[COURSE_ja.md](COURSE_ja.md) を参照(演習見出しの「第N回」は同講義の回)。

前提: `pip install pyscf-cli` 済み、`pyscf-cli examples all` でサンプル分子を取得。

---

## 1. 化学結合はなぜできるか — H₂ の PES スキャン(30分)

```bash
pyscf-cli energy h2.xyz --pes --rmin 0.4 --rmax 3.0 --npts 27 --basis 6-31g
```

- 極小の位置(平衡結合長)と実験値 0.741 Å を比較させる
- **問い:** R → ∞ で RHF のエネルギーはどこに向かうか? H原子2個分
  (`pyscf-cli energy h.xyz --spin 1 --method uhf` × 2)と比較すると?
- 大きな R で「SCF not converged」の警告が出る点こそ教材:
  **制限付き HF は等方解離を記述できない**(イオン項の混入)。
  `--method uhf` で再スキャンし、両者の解離挙動の違いを議論する

## 2. O₂ はなぜ磁石にくっつくか — スピン状態(30分)

```bash
pyscf-cli energy o2.xyz --basis 6-31g                       # 一重項を仮定
pyscf-cli energy o2.xyz --basis 6-31g --spin 2 --method uhf # 三重項
```

- どちらが低エネルギーか? → 基底状態は三重項(常磁性の起源)
- **⟨S²⟩ の読み方**を教える: 2.0 に近ければきれいな三重項、
  大きくずれていればスピン汚染
- 発展: `--json` で両エネルギーを取り出し、一重項–三重項ギャップを
  eV と kJ/mol で報告させる

## 3. 電子配置と励起状態 — Be 原子の完全コース(60–90分)

この一連の流れは「軌道エネルギー差 ≠ 励起エネルギー」を体感させる教材です。

```bash
# (a) 基底状態 (2s)² と MO 準位
pyscf-cli energy be.xyz --basis cc-pvdz

# (b) 凍結軌道で 2s→2p 昇位: K 積分と一重項/三重項分裂まで表示される
pyscf-cli energy be.xyz --basis cc-pvdz --fixed-occ-decomp

# (c) 軌道緩和を許す Δ-SCF: 配置 (1s)²(2s)¹(2p)¹ を明示指定
pyscf-cli energy be.xyz --basis cc-pvdz --method uhf --occ-alpha 1,3

# (d) 三重項は --spin で
pyscf-cli energy be.xyz --basis cc-pvdz --spin 2 --method uhf
```

- (b) の出力にある Triplet(Δ−K)/ Singlet(Δ+K)を実験値
  (³P 2.72 eV, ¹P 5.28 eV)と比較
- (c) の ⟨S²⟩ = 1.0 の意味を問う → mₛ=0 行列式は一重項と三重項の
  50:50 混合。**スピン純化則** E(¹P) ≈ 2E(mₛ=0) − E(³P) を黒板で導出し、
  (c)(d) の数値で検証させる
- **問い:** 軌道エネルギー差(≈10 eV)より励起エネルギーがずっと小さい
  のはなぜか(J による安定化、軌道緩和)

## 4. 分子は振動する — H₂O の基準振動(45分)

```bash
pyscf-cli relax h2o.xyz --basis 6-31g          # まず構造最適化
pyscf-cli vib h2o-finish.xyz --basis 6-31g     # 最適化構造で振動解析
pyscf-cli vibmovie h2o-finish.xyz --basis 6-31g
```

- 3N−6 = 3 モード(変角・対称伸縮・逆対称伸縮)を GIF で同定させる
- 実験値(1595, 3657, 3756 cm⁻¹)との比較 → 調和近似と基底の限界
- **虚振動の教材化:** 最適化前の歪んだ構造や直線 H₂O を `vib` にかけ、
  虚振動=鞍点の意味を議論(CLI が警告を出します)
- CO₂ で 3N−5 = 4 モード(直線分子)も確認

## 5. 反応の熱力学 — 水の生成反応(45分)

H₂ + ½ O₂ → H₂O の ΔH, ΔG を計算:

```bash
pyscf-cli thermo h2.xyz  --basis 6-31g** --json h2.json
pyscf-cli thermo o2.xyz  --basis 6-31g** --spin 2 --method uhf --json o2.json
pyscf-cli thermo h2o.xyz --basis 6-31g** --json h2o.json
```

- JSON から H_tot / G_tot を取り出して差を取らせる(採点も自動化可能)
- 実験値 ΔH°f = −241.8 kJ/mol(気相)との比較
- **問い:** なぜ HF ではズレるのか → 相関エネルギー。
  `--theory mp2` は `energy` で試せる(thermo は SCF/DFT のみ)

## 6. π 共役と HOMO/LUMO — ベンゼンの DOS と軌道(45分)

```bash
pyscf-cli dos benzene.xyz --basis 6-31g --element-pdos --align homo
pyscf-cli orbitals benzene.xyz --basis 6-31g --homo --lumo
```

- DOS 図で HOMO 付近が C の p 軌道由来(π)であることを読み取らせる
- cube を VESTA で開き、HOMO/LUMO の節面を観察
  (VESTA で軸表示が2組出たら: cube は単独で File > Open で開く)
- 発展: ナフタレンと比較して HOMO–LUMO ギャップの縮小(共役拡大)を確認

## 7. 基底関数系と理論レベルの収束(60分・レポート向け)

```bash
for b in sto-3g 6-31g 6-31g** cc-pvdz cc-pvtz; do
  pyscf-cli energy h2o.xyz --basis $b --json h2o_$b.json
done
for t in scf mp2 ccsd ccsd_t; do
  pyscf-cli energy h2o.xyz --basis cc-pvdz --theory $t --json h2o_$t.json
done
```

- 基底の系統的改善と変分原理(エネルギーは単調に下がるか?)
- 相関エネルギーの定義 E_corr = E − E_HF を計算させる
- **問い:** 「基底を良くする」と「理論を良くする」はどちらが効くか?
  計算時間はどうスケールするか(CCSD(T) は N⁷!)

## 8. 卒業試験 — CLI から PySCF へ(自習)

```bash
pyscf-cli energy h2o.xyz --theory mp2 --basis cc-pvdz --dry-run > my_first_pyscf.py
python3 my_first_pyscf.py
```

- 生成されたスクリプトを読み、各行が CLI のどのオプションに対応するか
  説明させる
- スクリプトを改造して(例: 基底を振る for ループ)、CLI では
  できないことを1つやらせる — これが本講義の到達点

## 9. 交換エネルギーと自己相互作用(第3・9回、45分)

水素原子1個 — 電子1個なのに「電子間相互作用」の項はどうなる?

```bash
pyscf-cli energy h.xyz --spin 1 --method uhf --basis 6-31g --decompose-total-energy
pyscf-cli energy he.xyz --basis 6-31g --decompose-total-energy
```

- H では **Hartree (U) と Exchange (J) が符号違いで完全に打ち消し合う**
  ことを確認 → 交換項の役割の一つは「自分で自分に反発する」偽の項
  (自己相互作用)の除去である
- He では U+J ≈ 30 eV(電子間反発)。第3回の変分見積もり・第4回の
  1次摂動(スライドの値)と並べて表にする
- **発展:** この自己相互作用の打ち消しが近似 DFT では不完全なこと
  (SIE)が、固体のバンドギャップ過小評価につながる、という現代の
  研究トピックへ接続できる

## 10. He の数値ラダー — 「電子相関」に名前を付ける(第3・4回、45分)

```bash
pyscf-cli energy he.xyz --basis cc-pvdz                 # HF
pyscf-cli energy he.xyz --basis cc-pvdz --theory ccsd   # ほぼ厳密(2電子系)
pyscf-cli energy he.xyz --basis aug-cc-pvqz --theory ccsd
```

- 講義の数値ラダー「反発無視 −108.8 → 1次摂動 −74.8 → HF −77.9 →
  実験 −79.0 eV」に CCSD の値を書き足させる
- E_corr = E(CCSD) − E(HF) を計算し、「摂動でも変分でも埋まらない残り」
  に**電子相関**という名前が付いていることを体感させる
- 基底を上げると実験値にどこまで迫れるか?(基底極限の概念)

## 11. C 原子のフント則と high/low spin(第7・9回、30分)

```bash
pyscf-cli energy c.xyz --basis 6-31g --spin 2 --method uhf   # 2p² 三重項(基底状態)
pyscf-cli energy c.xyz --basis 6-31g --spin 4 --method uhf   # 2s¹2p³ 五重項
```

- どちらが低い? spin 4 は交換安定化(同スピン対の増加)を得るが
  2s→2p の昇位コストを払う — フント則と昇位コストの綱引きを数値で
- `--decompose-total-energy` で Exchange (J) 項だけを比較させると
  交換安定化が直接見える
- d⁴ 錯体の high/low spin の手計算課題(第9回)の数値版

## 12. NaCl は気相ではイオン化しない — 固体への橋(第14回、45分)

```bash
pyscf-cli energy na.xyz --spin 1 --method uhf --basis 6-31g --json na.json
pyscf-cli energy na.xyz --spin 0 --charge 1 --basis 6-31g --json na+.json
pyscf-cli energy cl.xyz --spin 1 --method uhf --basis 6-31g --json cl.json
pyscf-cli energy cl.xyz --spin 0 --charge -1 --basis "6-31+g" --json cl-.json
```

- イオン化エネルギー I(Na) と電子親和力 A(Cl) を差分から計算 →
  **A − I < 0**: 孤立した Na + Cl は電子を渡すだけ損
- ではなぜ食塩は Na⁺Cl⁻ なのか? → 失った分を取り返すのが結晶中の
  **Madelung エネルギー**(第13回)— 分子計算だけで固体物理の入口に立てる
- 注意: アニオンには diffuse 基底(`6-31+g`)が必要な点も教材になる

## 13. 結合性・反結合性を「見る」— COOP/COHP 解析(第10・11回、45分)

```bash
pyscf-cli dos h2o.xyz --basis 6-31g --coop --cohp --align homo
pyscf-cli dos oh.xyz --basis 6-31g --spin 1 --method uhf --coop --align homo
```

- OP_/HP_ プロットで、各準位が O-H 結合に対して**結合性(+)か
  反結合性(−)か**を読み取らせる(DOS だけでは分からない情報)
- ICOOP(積分値)は結合次数の近似指標。H₂O と OH ラジカルの O-H を比較
- OH のスピン分解 COOP では α/β の寄与の違い(不対電子の効果)が見える
- 注意: COOP/COHP は**元素ペア**解析なので、O₂ のような単一元素分子には
  適用できない(異核分子を選ぶこと)
- 固体物性で頻出する COHP 解析(LOBSTER 等)の分子版として紹介できる

## 14. PubChem から一気通貫 — 実分子パイプライン(第10回、90分)

Molcalc 的な体験を CLI で完結させる、実習の集大成:

```bash
# 1. PubChem で好きな分子を検索し、3D Conformer を SDF でダウンロード
# 2. 変換 → 最適化 → 各種解析
pyscf-cli convert SDF_aspirin.sdf          # → XYZ_aspirin.xyz
pyscf-cli relax XYZ_aspirin.xyz --basis sto-3g
pyscf-cli energy XYZ_aspirin-finish.xyz --basis 6-31g
pyscf-cli dos XYZ_aspirin-finish.xyz --element-pdos --align homo
pyscf-cli orbitals XYZ_aspirin-finish.xyz --homo --lumo
pyscf-cli vib XYZ_aspirin-finish.xyz --basis sto-3g
```

- 学生が**自分で選んだ分子**で全機能を使う(提出課題向き)
- 注意点も教材: 大きい分子ほど計算時間が急増(事前警告が出る)、
  最適化前後で構造がどう変わったか VESTA で比較、など

---

## 運用メモ

- **環境:** 教室では Google Colab が最も確実(インストール不要、
  Windows 問題も回避)。`examples/colab_quickstart.ipynb` を配布してください
- **計算時間の目安:** ここに挙げた計算はすべて1分以内(多くは数秒)。
  CCSD(T)/cc-pVTZ 以上や10原子超の分子は授業内では避けるのが無難です
  (CLI が事前警告を出します)
- **採点:** `--json` + 終了コード(0=成功, 2=入力エラー, 3=未収束)で
  スクリプト採点が組めます
- 誤った入力への丁寧なエラーメッセージも「わざと間違えさせる」教材として
  使えます(例: O₂ を `--spin 1` で計算させてみる)

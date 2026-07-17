# pyscf-cli 演習アイデア集(教員向け)

量子化学の講義・学生実験で pyscf-cli をそのまま使うための演習例です。
各演習に、コマンド、学生への問いかけ、所要時間の目安を付けています。
計算はすべてノートPC(または Google Colab)で数秒〜数分で終わるよう、
小さな分子と基底で設計しています。

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

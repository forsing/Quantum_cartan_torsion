# Quantum_cartan_torsion



import random
import numpy as np

random.seed(39)
np.random.seed(39)

from pathlib import Path
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



FILES = (
    Path("/data/loto7_4718_k90.csv"),
    Path("/data/loto7_4718_k90_loto_2981.csv"),
    Path("/data/loto7_4718_k90_loto_plus_1737.csv"),
)



NUM_BALLS = 39
DRAW_SIZE = 7
N_QUBITS = 6
N_STATES = 2 ** N_QUBITS


def load_draws(csv_path: Path) -> np.ndarray:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.replace(";", ",").split(",") if p.strip()]
            nums = [int(x) for x in parts[:DRAW_SIZE]]
            if len(nums) != DRAW_SIZE:
                raise ValueError(f"Neispravan red u {csv_path}: {line}")
            if any(n < 1 or n > NUM_BALLS for n in nums) or len(set(nums)) != DRAW_SIZE:
                raise ValueError(f"Nelegalna kombinacija u {csv_path}: {nums}")
            rows.append(sorted(nums))
    if len(rows) < 100:
        raise ValueError(f"Premalo kola u {csv_path}: {len(rows)}")
    return np.array(rows, dtype=np.int32)


def draws_to_binary_matrix(draws: np.ndarray) -> np.ndarray:
    t_len = len(draws)
    mat = np.zeros((t_len, NUM_BALLS), dtype=np.float64)
    rows_idx = np.arange(t_len)[:, None]
    mat[rows_idx, draws - 1] = 1.0
    return mat


def compute_causal_cartan_torsion_signals(
    binary_mat: np.ndarray, memory_decay: float = 0.96
) -> tuple[np.ndarray, np.ndarray]:
    t_len, n_balls = binary_mat.shape
    torsion_field = np.zeros((t_len, n_balls), dtype=np.float64)
    kozyrev_flux = np.zeros(t_len, dtype=np.float64)
    trans_mat = np.zeros((n_balls, n_balls), dtype=np.float64)

    for t in range(t_len):
        x_curr = binary_mat[t]
        if t == 0:
            kozyrev_flux[t] = 0.0
            torsion_field[t] = np.zeros(n_balls, dtype=np.float64)
            continue

        x_prev = binary_mat[t - 1]
        kozyrev_flux[t] = float(np.sum(np.abs(x_curr - x_prev))) / (2.0 * float(DRAW_SIZE))

        trans_mat = memory_decay * trans_mat + np.outer(x_prev, x_curr)
        cartan_skew = 0.5 * (trans_mat - trans_mat.T)
        norm_factor = float(np.sum(np.abs(cartan_skew))) + 1e-12
        cartan_skew_norm = cartan_skew / norm_factor

        torsion_field[t] = x_curr @ cartan_skew_norm

    return torsion_field, kozyrev_flux


def encode_torsion_to_64_amplitudes(
    x_curr: np.ndarray,
    torsion_vec: np.ndarray,
    flux_val: float,
    alpha_torsion: float,
    beta_state: float,
) -> np.ndarray:
    eff_alpha = alpha_torsion * (0.5 + flux_val)
    logits_39 = beta_state * x_curr + eff_alpha * torsion_vec
    logits_39 = np.clip(logits_39, -8.0, 8.0)

    exp_39 = np.exp(logits_39 - float(np.max(logits_39)))
    probs_39 = exp_39 / float(np.sum(exp_39))

    active_share = float(NUM_BALLS) / float(N_STATES)
    tail_len = N_STATES - NUM_BALLS

    amp_64 = np.zeros(N_STATES, dtype=np.float64)
    amp_64[:NUM_BALLS] = np.sqrt(active_share * probs_39)
    if tail_len > 0:
        amp_64[NUM_BALLS:] = np.sqrt((1.0 - active_share) / float(tail_len))

    amp_64 /= float(np.linalg.norm(amp_64))
    return amp_64


def quantum_cartan_spin_evolution(
    amp_64: np.ndarray, spin_params: np.ndarray, flux_val: float
) -> np.ndarray:
    qc = QuantumCircuit(N_QUBITS)
    qc.initialize(amp_64.tolist(), list(range(N_QUBITS)))

    idx = 0
    flux_scale = 0.5 + flux_val
    for _ in range(2):
        for q in range(N_QUBITS):
            q_next = (q + 1) % N_QUBITS
            phi = float(spin_params[idx]) * flux_scale
            idx += 1
            theta = float(spin_params[idx])
            idx += 1

            qc.cx(q, q_next)
            qc.rz(phi, q_next)
            qc.ry(-phi, q)
            qc.cx(q, q_next)
            qc.ry(theta, q)

    sv = Statevector.from_instruction(qc)
    probs_64 = np.abs(sv.data) ** 2

    dist_39 = probs_64[:NUM_BALLS].copy()
    dist_39 = np.maximum(dist_39, 1e-12)
    dist_39 /= float(np.sum(dist_39))
    return dist_39


def predict_step_distribution(
    x_curr: np.ndarray,
    torsion_vec: np.ndarray,
    flux_val: float,
    full_params: np.ndarray,
) -> np.ndarray:
    alpha_torsion, beta_state = full_params[:2]
    spin_params = full_params[2:]

    amp_64 = encode_torsion_to_64_amplitudes(
        x_curr=x_curr,
        torsion_vec=torsion_vec,
        flux_val=flux_val,
        alpha_torsion=float(alpha_torsion),
        beta_state=float(beta_state),
    )
    return quantum_cartan_spin_evolution(amp_64, spin_params, flux_val)


def train_cartan_quantum_model(
    x_seq: np.ndarray,
    torsion_seq: np.ndarray,
    flux_seq: np.ndarray,
    target_binary_seq: np.ndarray,
    reg_lambda: float = 1.5e-3,
    max_iter: int = 45,
) -> np.ndarray:
    n_samples = len(x_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)

    step = max(1, n_samples // 180)
    sub_x = x_seq[::step]
    sub_t = torsion_seq[::step]
    sub_f = flux_seq[::step]
    sub_tgt = target_dist[::step]
    sub_len = len(sub_x)

    n_spin_params = 2 * N_QUBITS * 2
    rng = np.random.RandomState(39)
    init_head = np.array([0.50, -0.10], dtype=np.float64)
    init_spin = rng.uniform(-0.20, 0.20, size=n_spin_params)
    init_params = np.concatenate([init_head, init_spin])

    def loss_fn(params: np.ndarray) -> float:
        total_loss = 0.0
        for i in range(sub_len):
            pred_dist = predict_step_distribution(sub_x[i], sub_t[i], float(sub_f[i]), params)
            total_loss -= float(np.sum(sub_tgt[i] * np.log(pred_dist)))
        total_loss /= float(sub_len)
        total_loss += reg_lambda * float(np.sum(params[2:] ** 2))
        return float(total_loss)

    res = minimize(
        loss_fn,
        init_params,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": 1e-5},
    )
    return res.x


def evaluate_distribution(
    x_seq: np.ndarray,
    torsion_seq: np.ndarray,
    flux_seq: np.ndarray,
    target_binary_seq: np.ndarray,
    params: np.ndarray,
) -> tuple[float, float]:
    n_samples = len(x_seq)
    target_dist = target_binary_seq / float(DRAW_SIZE)
    nll = 0.0
    hits = 0.0
    for i in range(n_samples):
        pred_dist = predict_step_distribution(x_seq[i], torsion_seq[i], float(flux_seq[i]), params)
        nll -= float(np.sum(target_dist[i] * np.log(pred_dist)))
        top7 = np.argsort(pred_dist)[-DRAW_SIZE:]
        actual = np.flatnonzero(target_binary_seq[i] > 0.5)
        hits += float(len(set(top7.tolist()).intersection(actual.tolist())))
    return nll / float(n_samples), hits / float(n_samples)


def select_next_combination(dist_39: np.ndarray) -> list[int]:
    if dist_39.shape != (NUM_BALLS,) or not np.all(np.isfinite(dist_39)):
        raise ValueError("Neispravna izlazna raspodela modela.")
    if abs(float(np.sum(dist_39)) - 1.0) > 1e-5:
        raise ValueError("Raspodela nije normalizovana.")
    top7_idx = np.argsort(dist_39)[-DRAW_SIZE:]
    combo = sorted(int(i + 1) for i in top7_idx)
    if len(combo) != DRAW_SIZE or len(set(combo)) != DRAW_SIZE:
        raise ValueError("Nelegalna izlazna kombinacija.")
    return combo


def run_pipeline_for_csv(csv_path: Path) -> None:
    draws = load_draws(csv_path)
    binary_mat = draws_to_binary_matrix(draws)
    t_len = len(binary_mat)

    n_trans = t_len - 1
    train_end = int(n_trans * 0.70)
    val_end = int(n_trans * 0.85)

    candidate_decays = (0.94, 0.985)

    best_decay = next(iter(candidate_decays))
    best_val_nll = float("inf")
    best_val_hits = 0.0
    best_test_nll = float("inf")
    best_test_hits = 0.0

    for decay in candidate_decays:
        torsion_all, flux_all = compute_causal_cartan_torsion_signals(
            binary_mat, memory_decay=decay
        )

        x_train = binary_mat[1:train_end]
        t_train = torsion_all[1:train_end]
        f_train = flux_all[1:train_end]
        y_train = binary_mat[2 : train_end + 1]

        x_val = binary_mat[train_end:val_end]
        t_val = torsion_all[train_end:val_end]
        f_val = flux_all[train_end:val_end]
        y_val = binary_mat[train_end + 1 : val_end + 1]

        params_cv = train_cartan_quantum_model(
            x_train, t_train, f_train, y_train, max_iter=35
        )
        val_nll, val_hits = evaluate_distribution(x_val, t_val, f_val, y_val, params_cv)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_val_hits = val_hits
            best_decay = decay

            x_test = binary_mat[val_end:n_trans]
            t_test = torsion_all[val_end:n_trans]
            f_test = flux_all[val_end:n_trans]
            y_test = binary_mat[val_end + 1 : n_trans + 1]
            best_test_nll, best_test_hits = evaluate_distribution(
                x_test, t_test, f_test, y_test, params_cv
            )

    torsion_final, flux_final = compute_causal_cartan_torsion_signals(
        binary_mat, memory_decay=best_decay
    )

    x_full = binary_mat[1:n_trans]
    t_full = torsion_final[1:n_trans]
    f_full = flux_final[1:n_trans]
    y_full = binary_mat[2 : n_trans + 1]
    final_params = train_cartan_quantum_model(
        x_full, t_full, f_full, y_full, max_iter=50
    )

    next_dist = predict_step_distribution(
        binary_mat[-1], torsion_final[-1], float(flux_final[-1]), final_params
    )
    next_combo = select_next_combination(next_dist)

    print(f"CSV: {csv_path.name} (kola: {t_len})")
    print(f"Validacija -> NLL: {best_val_nll:.4f} | Prosek pogodaka: {best_val_hits:.3f}")
    print(f"Test       -> NLL: {best_test_nll:.4f} | Prosek pogodaka: {best_test_hits:.3f}")
    print(f"NEXT       -> {next_combo}")
    print("-" * 52)





def main() -> None:
    print("Metoda: Einstein-Cartan Antisymmetric Torsion + Quantum Spin (6 qubita)")
    print("Posebnost: Kauzalni tenzor torzije prelaza i antisimetricna sprega spina")
    print("=" * 52)
    for csv_path in FILES:
        run_pipeline_for_csv(csv_path)


if __name__ == "__main__":
    main()



"""
Metoda: Einstein-Cartan Antisymmetric Torsion + Quantum Spin (6 qubita)
Posebnost: Kauzalni tenzor torzije prelaza i antisimetricna sprega spina
====================================================

CSV: loto7_4718_k90.csv (kola: 4718)
Validacija -> NLL: 3.6679 | Prosek pogodaka: 1.227
Test       -> NLL: 3.6697 | Prosek pogodaka: 1.199
NEXT       -> [3, x, 9, y, 12, z, 33]
----------------------------------------------------

CSV: loto7_4718_k90_loto_2981.csv (kola: 2981)
Validacija -> NLL: 3.6653 | Prosek pogodaka: 1.244
Test       -> NLL: 3.6676 | Prosek pogodaka: 1.210
NEXT       -> [8, x, 22, y, 24, z, 39]
----------------------------------------------------

CSV: loto7_4718_k90_loto_plus_1737.csv (kola: 1737)
Validacija -> NLL: 3.6654 | Prosek pogodaka: 1.269
Test       -> NLL: 3.6641 | Prosek pogodaka: 1.268
NEXT       -> [15, x, 31, y, 34, z, 39]
----------------------------------------------------
"""



"""
Russian Torsion Physics: New Force of Nature? 
Nikolai Aleksandrovich Kozyrev
French mathematician Élie Cartan 
Einstein-Cartan Theory 
Akimov and Shipov 
Albert-Viktor Veynik 

torziona fizika / Einstein-Cartan teorija / Kozyrevljeva kauzalna mehanika


1. Einstein-Cartan teorija (Élie Cartan): Standardna matematička ekstenzija opšte relativnosti gde afina koneksija nije simetrična, već sadrži antisimetrični deo — tenzor torzije \(S^\alpha_{\mu\nu} = \Gamma^\alpha_{[\mu\nu]}\), koji povezuje geometriju prostora sa unutrašnjim ugaonim momentom (spinom).
2. Kauzalna mehanika (Nikolai Kozyrev): Koncept prema kome je vreme usmereno i asimetrično (uzrok i posledica su uvek strogo razdvojeni u prostoru i vremenu, \(\delta t > 0\)), pri čemu se promena "gustine toka vremena" opisuje kroz asimetriju između prošlog i budućeg stanja.
3. Radovi Akimova, Shipova i Veynika (članak Tima Venture): Pokušaji iz 1980-ih i 1990-ih da se Einstein-Cartan torzija i Kozyrevljeve ideje prošire na makrofizička "torziona polja" i hronalne fenomene, koje je zvanična nauka odbacila kao nedokazane.

Strogi matematički formalizam Élie Cartana i Kozyrevljeve kauzalne asimetrije (bez pseudonaučnih tvrdnji o "trenutnom prenosu".

• Antisimetrični tenzor torzije (Cartan) za modelovanje usmerenih prelaza: U običnim korelacionim matricama veza između broja \(i\) i broja \(j\) je simetrična (\(C_{ij} = C_{ji}\)), pa gubi informaciju o smeru kroz vreme. Cartanova antisimetrična koneksija \(T_{ij} = \frac{1}{2}(P(i_t \to j_{t+1}) - P(j_t \to i_{t+1}))\) direktno meri usmereni spin / torziju prelaza između dva uzastopna kola, odvajajući čisti vremenski redosled od statičke učestalosti.
• Kauzalna asimetrija (Kozyrev) kao lokalna gustina promena: U skladu sa Kozyrevljevom kauzalnom asimetrijom (\(\delta t > 0\)), brzina promene stanja sistema između kola \(t-1\) i \(t\) definiše lokalni fazni/rotacioni ugao bez ikakvog uvida u \(t+1\).
• Prirodna veza sa kvantnim spinom (Qiskit, 6 qubita: U Einstein-Cartan formalizmu torzija je direktno spregnuta sa kvantnim spinom. U Qiskit simulatoru (6 qubita = 64 stanja), antisimetrični torzioni operator (rotacije sa suprotnim fazama \(R_{yz} - R_{zy}\) preko spregnutih qubita) deluje nad amplitudama trenutnog stanja i proizvodi regresiju raspodele za sledeće kolo bez klasifikatora i bez kozmetičkih filtera.

Einstein-Cartan antisimetrični tenzor torzija prelaza + kvantni spregnuti spin na 6 qubita 
"""

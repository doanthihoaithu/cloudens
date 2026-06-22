"""
Overfit-a-Single-Batch tests for every anomaly detection model wrapper.

Strategy:
  Create one fixed synthetic batch (inputs [B,T,N,F], labels [B,N,F]) and
  repeatedly run the model's own training step on that same batch.  A healthy
  model should be able to memorise the batch — its loss must drop by at least
  PASS_RATIO within N_STEPS gradient steps.  If it cannot, the forward pass,
  loss, or backward pass has a bug.

Pass criterion: final_loss / initial_loss < PASS_RATIO  (default: 0.5)
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F

# ── Synthetic-data configuration ──────────────────────────────────────────────
B        = 4      # batch size
T        = 6      # slide window length
N        = 5      # number of nodes / variables
F_DIM    = 2      # features per node
N_STEPS  = 200    # gradient steps per test
PASS_RATIO = 0.5  # final/initial loss threshold for PASS
DEVICE   = 'cpu'


def make_batch():
    """Return one fixed (inputs, labels) batch on DEVICE."""
    torch.manual_seed(0)
    inputs = torch.rand(B, T, N, F_DIM, device=DEVICE)   # [B, T, N, F]
    labels = torch.rand(B, N, F_DIM, device=DEVICE)       # [B, N, F]
    return inputs, labels


def _report(name: str, losses: list) -> bool:
    initial, final = losses[0], losses[-1]
    ratio = final / (initial + 1e-9)
    passed = ratio < PASS_RATIO
    tag = 'PASS' if passed else 'FAIL'
    print(f'  [{tag}] {name:14s}  loss: {initial:.6f} → {final:.6f}  (ratio {ratio:.3f})')
    return passed


# ── GDN ───────────────────────────────────────────────────────────────────────

def test_gdn() -> bool:
    """
    GDN: learnable graph + graph-attention over GRU-encoded histories.
    Forward: model(inputs) → [B, N, F].  Loss: MSE.
    """
    from model_wrappers.GDNWrapper import GDNWrapper

    w = GDNWrapper(num_nodes=N, node_features=F_DIM, slide_win=T,
                   embed_dim=16, hidden_dim=32, topk=min(3, N - 1),
                   device=DEVICE)
    inputs, labels = make_batch()

    losses = []
    for _ in range(N_STEPS):
        w.model.train()
        pred = w.model(inputs)
        loss = w.loss_fn(pred, labels)
        w.optimizer.zero_grad()
        loss.backward()
        w.optimizer.step()
        losses.append(loss.item())

    return _report('GDN', losses)


# ── TranAD ────────────────────────────────────────────────────────────────────

def test_tranad() -> bool:
    """
    TranAD: Transformer with two decoders + focus-weighted adversarial loss.
    Forward: model(inputs) → (pred1, pred2).
    Loss: MSE(pred1, labels) + focus-weighted MSE(pred2, labels).
    """
    from model_wrappers.TranADWrapper import TranADWrapper

    w = TranADWrapper(num_nodes=N, node_features=F_DIM, slide_win=T,
                      d_model=32, nhead=2, n_layers=1,
                      device=DEVICE)
    inputs, labels = make_batch()

    losses = []
    for _ in range(N_STEPS):
        w.model.train()
        pred1, pred2 = w.model(inputs)
        loss = w._compute_loss(pred1, pred2, labels)
        w.optimizer.zero_grad()
        loss.backward()
        w.optimizer.step()
        # Track plain MSE of the primary decoder as the indicator
        losses.append(F.mse_loss(pred1.detach(), labels).item())

    return _report('TranAD', losses)


# ── OmniAnomaly ───────────────────────────────────────────────────────────────

def test_omni_anomaly() -> bool:
    """
    OmniAnomaly: VAE with GRU encoder/decoder (ELBO = recon + β·KL).
    Forward: model(inputs) → (pred, mu, logvar).
    Tracks reconstruction MSE; β=0.01 so KL regularisation is minimal.
    """
    from model_wrappers.OmniAnomalyWrapper import OmniAnomalyWrapper

    w = OmniAnomalyWrapper(num_nodes=N, node_features=F_DIM, slide_win=T,
                           hidden_dim=32, latent_dim=8, n_layers=2,
                           beta=0.01,   # near-zero KL weight lets reconstruction dominate
                           device=DEVICE)
    inputs, labels = make_batch()

    # VAE: KL regulariser slows convergence → allow extra steps
    losses = []
    for _ in range(N_STEPS * 2):
        w.model.train()
        pred, mu, logvar = w.model(inputs)
        loss = w._elbo_loss(pred, labels, mu, logvar)
        w.optimizer.zero_grad()
        loss.backward()
        w.optimizer.step()
        losses.append(w.recon_loss_fn(pred.detach(), labels).item())

    return _report('OmniAnomaly', losses)


# ── USAD ──────────────────────────────────────────────────────────────────────

def test_usad() -> bool:
    """
    USAD: shared GRU encoder + two adversarial decoders, two optimisers.
    Forward: model(inputs) → (pred1, pred2, pred12).
    Uses alpha=1.0 (pure reconstruction, no adversarial term) so both
    decoders independently minimise MSE — making overfit straightforward.
    """
    from model_wrappers.USADWrapper import USADWrapper

    w = USADWrapper(num_nodes=N, node_features=F_DIM, slide_win=T,
                    hidden_dim=32, latent_dim=16, n_layers=1,
                    device=DEVICE)
    inputs, labels = make_batch()

    alpha = 1.0   # epoch-1 schedule: pure reconstruction, no adversarial shift
    losses = []

    for _ in range(N_STEPS):
        w.model.train()

        # AE1 update (encoder + decoder1)
        pred1, pred2, pred12 = w.model(inputs)
        loss_ae1, _ = w._step_losses(pred1, pred2, pred12, labels, alpha)
        w.optimizer1.zero_grad()
        loss_ae1.backward()
        w.optimizer1.step()

        # AE2 update (encoder + decoder2)
        pred1, pred2, pred12 = w.model(inputs)
        _, loss_ae2 = w._step_losses(pred1, pred2, pred12, labels, alpha)
        w.optimizer2.zero_grad()
        loss_ae2.backward()
        w.optimizer2.step()

        losses.append(loss_ae1.item())   # L_AE1 = MSE(pred1, y) when alpha=1

    return _report('USAD', losses)


# ── MTAD-GAT ──────────────────────────────────────────────────────────────────

def test_mtad_gat() -> bool:
    """
    MTAD-GAT: dual-attention (feature-GAT + time-GAT) + GRU + joint loss.
    Forward: model(inputs) → (forecast [B,N,F], recon [B,T,N,F]).
    Loss: MSE(forecast, labels) + MSE(recon, inputs). Tracks forecast MSE.
    """
    from model_wrappers.MTADGATWrapper import MTADGATWrapper

    w = MTADGATWrapper(num_nodes=N, node_features=F_DIM, slide_win=T,
                       d_model=16, nhead=1, gru_hidden=32,
                       device=DEVICE)
    inputs, labels = make_batch()

    losses = []
    for _ in range(N_STEPS):
        w.model.train()
        forecast, recon = w.model(inputs)
        loss = w.loss_fn(forecast, labels) + w.loss_fn(recon, inputs)
        w.optimizer.zero_grad()
        loss.backward()
        w.optimizer.step()
        losses.append(w.loss_fn(forecast.detach(), labels).item())

    return _report('MTAD-GAT', losses)


# ── GST-Pro ───────────────────────────────────────────────────────────────────

def test_gst_pro() -> bool:
    """
    GST-Pro: learnable graph + spatial NCDE + temporal NCDE (Euler), masked L1.
    Forward: model(inputs) → (forecast [B,N,F], obs_mask [B,T,N,F]).
    Loss: masked L1(forecast, labels).
    """
    from model_wrappers.GSTPROWrapper import GSTPROWrapper

    w = GSTPROWrapper(num_nodes=N, node_features=F_DIM, slide_win=T,
                      embed_dim=16, hidden_dim=32, K=2,
                      device=DEVICE)
    inputs, labels = make_batch()

    losses = []
    for _ in range(N_STEPS):
        w.model.train()
        forecast, _ = w.model(inputs)
        loss = w._masked_l1(forecast, labels)
        w.optimizer.zero_grad()
        loss.backward()
        w.optimizer.step()
        losses.append(loss.item())

    return _report('GST-Pro', losses)


# ── Anomaly Transformer ───────────────────────────────────────────────────────

def test_anomaly_transformer() -> bool:
    """
    Anomaly Transformer: Anomaly-Attention (prior-GAT + series-GAT) + minimax.
    Forward: model(inputs) → (forecast [B,N,F], recon [B,T,N,F], assoc_disc).
    Minimax: Phase-1 maximises discrepancy (main params); Phase-2 minimises it (σ).
    Tracks forecast MSE as the indicator loss.
    """
    from model_wrappers.AnomalyTransformerWrapper import AnomalyTransformerWrapper

    w = AnomalyTransformerWrapper(num_nodes=N, node_features=F_DIM, slide_win=T,
                                   d_model=32, nhead=2, n_layers=2, lambda_=0.1,
                                   device=DEVICE)
    inputs, labels = make_batch()

    losses = []
    for _ in range(N_STEPS):
        losses.append(w._minimax_step(inputs, labels))

    return _report('AnomalyTransformer', losses)


# ── Orchestrator ──────────────────────────────────────────────────────────────

ALL_TESTS = [
    ('GDN',                  test_gdn),
    ('TranAD',               test_tranad),
    ('OmniAnomaly',          test_omni_anomaly),
    ('USAD',                 test_usad),
    ('MTAD-GAT',             test_mtad_gat),
    ('GST-Pro',              test_gst_pro),
    ('AnomalyTransformer',   test_anomaly_transformer),
]


def run_all_tests():
    sep = '=' * 57
    print(f'\n{sep}')
    print('  Overfit-a-Single-Batch Test Suite')
    print(f'  Batch: B={B}, T={T}, N={N}, F={F_DIM}  |  Steps: {N_STEPS}')
    print(f'  Pass criterion: final_loss / initial_loss < {PASS_RATIO}')
    print(f'{sep}\n')

    results = {}
    for name, fn in ALL_TESTS:
        print(f'Running {name} ...')
        try:
            results[name] = fn()
        except Exception as exc:
            print(f'  [ERROR] {name}: {exc}')
            import traceback; traceback.print_exc()
            results[name] = False
        print()

    n_pass = sum(results.values())
    n_total = len(results)
    print(sep)
    print('  Summary')
    print(sep)
    for name, passed in results.items():
        tag = 'PASS' if passed else 'FAIL'
        print(f'  {tag}  {name}')
    print(sep)
    print(f'  {n_pass}/{n_total} tests passed\n')
    return results


if __name__ == '__main__':
    run_all_tests()
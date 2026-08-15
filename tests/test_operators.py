import torch
from lgae_v3.types import make_graph_buffers
from lgae_v3.operators import actuation_operator, diagnostic_diffusion_operator, operator_discrepancy, DualOperatorState
from lgae_v3.kernels import FieldKernel, SparseFieldKernel
from lgae_v3.receipts import mutation_receipt, append_receipt

def test_dual_operators_are_row_stochastic():
    graph = make_graph_buffers(4, [(0, 1), (1, 2), (2, 3)], capacity=8)
    z = torch.randn(4, 6)
    pa = actuation_operator(graph)
    pd = diagnostic_diffusion_operator(z, k=3)
    assert torch.allclose(pa.sum(-1), torch.ones(4), atol=1e-6)
    assert torch.allclose(pd.sum(-1), torch.ones(4), atol=1e-6)
    assert operator_discrepancy(pa, pd) >= 0
    state = DualOperatorState(pa, pd)
    assert state.l_actuation.shape == (4, 4)
    assert state.l_diagnostic.shape == (4, 4)
    assert state.discrepancy() >= 0


def test_field_and_sparse_field_kernels():
    z = torch.randn(4, 8)
    p = torch.eye(4)
    fk = FieldKernel()
    z_next, gamma, rad, var = fk(z, p, 0.01)
    assert z_next.shape == (4, 8)
    assert gamma.shape == (4,)

    src = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 0], dtype=torch.long)
    pw = torch.ones(4, dtype=torch.float32)
    sfk = SparseFieldKernel()
    z_next2, g2, r2, v2 = sfk(z, src, dst, pw, 0.01)
    assert z_next2.shape == (4, 8)
    assert g2.shape == (4,)


def test_mutation_receipt(tmp_path):
    receipt = mutation_receipt({"status": "accepted", "step": 1})
    assert "sha256" in receipt
    assert receipt["schema"] == "LGAE_MUTATION_RECEIPT_V2"
    file_path = tmp_path / "receipts.jsonl"
    append_receipt(file_path, receipt)
    assert file_path.exists()
    assert len(file_path.read_text()) > 0


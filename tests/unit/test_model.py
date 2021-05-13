import torch

from time_series_forecasting.model import TimeSeriesForcasting, smape_loss


def test_smape_loss():
    target = torch.arange(1, 100)
    y_pred = target + 10

    loss = smape_loss(y_pred=y_pred, target=target).item()

    assert loss == 0.29728013277053833


def test_model():
    source = torch.rand(size=(32, 16, 9))
    target_in = torch.rand(size=(32, 16, 8))
    target_out = torch.rand(size=(32, 16, 1))

    ts = TimeSeriesForcasting(n_encoder_inputs=9, n_decoder_inputs=8)

    pred = ts((source, target_in))

    ts.training_step((source, target_in, target_out), batch_idx=1)

    assert pred.size() == torch.Size([32, 16, 1])

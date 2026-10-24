import numpy as np
import pytest
from PIL import Image
import json
from types import SimpleNamespace

from edgeface.preprocess import PREPROCESS, reject, softmax, tensor_from_rgb
from edgeface.config import Config
from raspberry_pi.inference import Classifier


def test_rgb_preprocessing_is_identical_for_pil_and_array():
    rgb = np.zeros((57, 103, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    array = tensor_from_rgb(rgb)
    np.testing.assert_array_equal(array, tensor_from_rgb(Image.fromarray(rgb)))
    assert array.shape == (1, 3, 224, 224) and array.dtype == np.float32
    assert float(array[0, 0, 0, 0]) == pytest.approx((1-0.485)/0.229)
    assert float(array[0, 2, 0, 0]) == pytest.approx((0-0.406)/0.225)


def test_unknown_explicit_and_uncertainty_rejection():
    classes = ["Unknown", "son", "another"]
    assert reject(np.array([0.8, 0.1, 0.1]), classes, 0.5)[0] == "Unknown"
    assert reject(np.array([0.3, 0.4, 0.3]), classes, 0.5)[0] == "Unknown"
    assert reject(np.array([0.05, 0.9, 0.05]), classes, 0.5)[0] == "son"


def test_runtime_override_rejects_confidence_without_changing_calibration(tmp_path, monkeypatch):
    (tmp_path/'classes.json').write_text(json.dumps(['Unknown', 'son', 'another']))
    (tmp_path/'preprocessing.json').write_text(json.dumps(PREPROCESS))
    calibration=json.dumps({'source': 'validation', 'threshold': .5833254400273035})
    (tmp_path/'threshold.json').write_text(calibration)
    (tmp_path/'face_classifier.onnx').write_bytes(b'unit-test-model-placeholder')
    session=SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name='input', shape=[1, 3, 224, 224], type='tensor(float)')],
        get_outputs=lambda: [SimpleNamespace(shape=[1, 3])],
        run=lambda *args: [np.log(np.array([[.05, .85, .10]]))])
    monkeypatch.setattr('raspberry_pi.inference.ort.InferenceSession', lambda *args, **kwargs: session)
    calibrated=Classifier(tmp_path)
    manual=Classifier(tmp_path, threshold_override=.9)
    image=np.zeros((224, 224, 3), dtype=np.uint8)
    assert calibrated.predict(image)[0] == 'son'
    assert manual.predict(image)[0] == 'Unknown'
    assert manual.threshold == .9 and manual.threshold_source == 'manual_override'
    assert manual.calibrated_threshold == calibrated.threshold
    assert manual.bundle_version == calibrated.bundle_version
    assert (tmp_path/'threshold.json').read_text() == calibration
    assert reject(np.array([.05, .9, .05]), ['Unknown', 'son', 'another'], .9)[0] == 'son'
    assert reject(np.array([.95, .03, .02]), ['Unknown', 'son', 'another'], .9)[0] == 'Unknown'


@pytest.mark.parametrize('value', ['-0.1', '1.1', 'nan', 'inf'])
def test_invalid_runtime_threshold_is_rejected(monkeypatch, value):
    monkeypatch.setenv('MQTT_PASSWORD', 'test-only')
    monkeypatch.setenv('REJECTION_THRESHOLD', value)
    with pytest.raises(ValueError, match='REJECTION_THRESHOLD'):
        Config.load()


def test_nonfinite_model_output_is_error_not_unknown():
    with pytest.raises(ValueError, match="non-finite"):
        softmax(np.array([[np.nan, 1]]))
    with pytest.raises(ValueError):
        tensor_from_rgb(np.zeros((20, 20, 3), np.uint8), {**PREPROCESS, "color": "BGR"})

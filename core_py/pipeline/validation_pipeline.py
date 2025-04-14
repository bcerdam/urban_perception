from core_py.utils import utils


def validate_model(weight_path, pp2_validation, m_w, m_t, similarity_threshold, inference_model, run_checkpoint_dir):
    utils.metrics(pp2_validation, inference_model, m_w, m_t, similarity_threshold, weight_path, None, run_checkpoint_dir)
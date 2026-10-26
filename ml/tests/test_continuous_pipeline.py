
import yaml
from kfp import compiler
from pipeline.continuous_pipeline import continuous_pipeline
from pipeline import schedule


def test_continuous_artifacts_secrets_and_gate(tmp_path):
    path = tmp_path/'continuous.yaml'
    compiler.Compiler().compile(continuous_pipeline, str(path))
    documents = list(yaml.safe_load_all(path.read_text()))
    spec = documents[0]
    tasks = spec['root']['dag']['tasks']
    assert set(tasks) == {'snapshot', 'train', 'evaluate', 'condition-1'}
    assert tasks['train']['inputs']['artifacts']['feedback']['taskOutputArtifact']['producerTask'] == 'snapshot'
    conditional = spec['components']['comp-condition-1']['dag']['tasks']
    assert set(conditional) == {'export', 'promote'}
    assert conditional['promote']['inputs']['artifacts']['model']['taskOutputArtifact']['producerTask'] == 'export'
    platform = documents[1]['platforms']['kubernetes']
    for name in ('exec-snapshot', 'exec-promote'):
        config = platform['deploymentSpec']['executors'][name]
        assert config['secretAsEnv'][0]['secretName'] == 'drawing-feedback'
    assert platform['pipelineConfig']['activeDeadlineSeconds'] == 7200


def test_schedule_replaces_only_managed_runs_after_upload(tmp_path):
    from types import SimpleNamespace as Obj
    events = []
    class Client:
        def create_recurring_run(self, **kwargs):
            assert kwargs['enabled'] is False
            assert kwargs['max_concurrency'] == 1
            assert kwargs['no_catchup'] is True
            events.append('create')
            return Obj(recurring_run_id='new')
        def list_recurring_runs(self, **kwargs):
            return Obj(recurring_runs=[Obj(display_name='drawing-classifier-daily', recurring_run_id='old'),
                                       Obj(display_name='unrelated', recurring_run_id='other')], next_page_token='')
        def disable_recurring_run(self, identity):
            events.append(('disable', identity))
        def enable_recurring_run(self, identity):
            events.append(('enable', identity))
    assert schedule.install(Client(), 'experiment', tmp_path/'pipeline.yaml', '0 0 18 * * *') == 'new'
    assert events == ['create', ('disable', 'old'), ('enable', 'new')]

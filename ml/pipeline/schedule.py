"""Install one daily recurrent run; replace prior managed schedules idempotently."""
import argparse
import os
from pathlib import Path
from kfp import Client, compiler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', required=True)
    parser.add_argument('--cron', default='0 0 18 * * *', help='KFP six-field UTC cron; default 03:00 JST')
    args = parser.parse_args()
    if not os.environ.get('DRAWING_TRAINING_IMAGE'):
        parser.error('set DRAWING_TRAINING_IMAGE to the newly built training image digest')
    from pipeline.continuous_pipeline import continuous_pipeline
    package = Path(__file__).with_name('drawing-classifier-continuous.yaml')
    compiler.Compiler().compile(continuous_pipeline, str(package))
    client = Client(host=args.host)
    experiment = client.create_experiment('drawing-classifier-continuous')
    recurring_id = install(client, experiment.experiment_id, package, args.cron)
    print(f'recurring_run_id={recurring_id}')


def install(client, experiment_id, package, cron):
    # Create disabled before changing the previous schedule, so upload failure leaves it intact.
    new = client.create_recurring_run(experiment_id=experiment_id,
                                     job_name='drawing-classifier-daily', pipeline_package_path=str(package),
                                     cron_expression=cron, max_concurrency=1, no_catchup=True,
                                     service_account='pipeline-runner', enabled=False,
                                     enable_caching=False)
    token = ''
    while True:
        page = client.list_recurring_runs(experiment_id=experiment_id, page_size=100, page_token=token)
        for run in page.recurring_runs or []:
            if run.display_name == 'drawing-classifier-daily' and run.recurring_run_id != new.recurring_run_id:
                client.disable_recurring_run(run.recurring_run_id)
        token = page.next_page_token
        if not token:
            break
    client.enable_recurring_run(new.recurring_run_id)
    return new.recurring_run_id


if __name__ == '__main__':
    main()

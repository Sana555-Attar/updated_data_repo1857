from platform import node
from clearml import Task
from clearml.automation import PipelineController

def pre_execute_callback_example(a_pipeline, a_node, current_param_override):
    # type (PipelineController, PipelineController.Node, dict) -> bool
    print('Cloning Task id={} with parameters: {}'.format(a_node.base_task_id, current_param_override))
    # if we want to skip this node (and subtree of this node) we return False
    # return True to continue DAG execution
    return True


def post_execute_callback_example(a_pipeline, a_node):
    # type (PipelineController, PipelineController.Node) -> None
    print('Completed Task id={}'.format(a_node.executed))
    # if we need the actual executed Task: Task.get_task(task_id=a_node.executed)
    return


def compare_metrics_and_publish_best(**kwargs):
    from clearml import OutputModel
    # Keep track of best node details
    current_best = dict()

    # For each incoming node, compare against current best
    for node_name, training_task_id in kwargs.items():
        # Get the original task based on the ID we got from the pipeline
        task = Task.get_task(task_id=training_task_id)
        accuracy = task.get_reported_scalars()['Performance']['Accuracy']['y'][0]
        model_id = task.get_models()['output'][0].id
        # Check if accuracy is better than current best, if so, overwrite current best
        if accuracy > current_best.get('accuracy', 0):
            current_best['accuracy'] = accuracy
            current_best['node_name'] = node_name
            current_best['model_id'] = model_id
            print(f"New current best model: {node_name}")

    # Print the final best model details and log it as an output model on this step
    print(f"Final best model: {current_best}")
    out_model = OutputModel(name="best_pipeline_model", base_model_id=current_best.get('model_id'), tags=['pipeline_winner'])
    out_model.publish()
    print(f'out_model.published {out_model.published}')


# Connecting ClearML with the current pipeline,
# from here on everything is logged automatically
pipe = PipelineController(
    name='sarcasm_detection_pipeline',
    project='sarcasm_detector',
    version='0.0.1'
)

pipe.set_default_execution_queue('CPU Queue')
pipe.add_parameter('C', [1,0.95,0.9])
#pipe.add_parameter('query_date', '2022-01-01')

pipe.add_step(
    name='data_preprocessing',
    base_task_project='sarcasm_detector',
    base_task_name='preprocess data'
)
pipe.add_step(
    name='train_sklearn',
    parents=['data_preprocessing'],
    base_task_project='sarcasm_detector',
    base_task_name='Sklearn Training',
    pre_execute_callback=pre_execute_callback_example,
    post_execute_callback=post_execute_callback_example
)
training_nodes = []
# Seeds should be pipeline arguments
# Don't change these when doing new run
for i, C in enumerate(pipe.get_parameters()['C']):
    node_name = f'model_training_{i}'
    training_nodes.append(node_name)
    pipe.add_step(
        name=node_name,
        parents=['data_preprocessing'],
        base_task_project='sarcasm_detector',
        base_task_name='Sklearn Training',
        parameter_override={'General/C': C}
    )

pipe.add_function_step(
    name='select_best_model',
    parents=training_nodes,
    function=compare_metrics_and_publish_best,
    function_kwargs={node_name: '${%s.id}' % node_name for node_name in training_nodes},
    monitor_models=["best_pipeline_model"]
)


# for debugging purposes use local jobs
#pipe.start_locally(run_pipeline_steps_locally=True)
# Starting the pipeline (in the background)
pipe.start()

print('Done!')

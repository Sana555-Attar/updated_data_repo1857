from clearml import Model, Task
import joblib

tsk = Task.init('sarcasm_detector','inference SKLEARN','inference')

sklearn_model_path = Model(model_id="baefc06449a0485f8a7f2887c934c798").get_local_copy()

sklearn_pipeline = joblib.load(sklearn_model_path)

args = {'sentences' : ["Coworkers At Bathroom Sink Locked In Tense Standoff Over Who Going To Wash Hands Longer","grandma jumps into buick for emergency birdseed run"]}
tsk.connect(args)

scores = sklearn_pipeline.predict_proba(args['sentences'])

print(f"Input Sentences :\n{args['sentences']}")

for idx, score in enumerate(scores):
    score = score[0]
    if score < 0.5:
        label = 'NORMAL'
        score = 1 - score
    else:
        label = 'SARCASTIC'
    print(f"Commnet: {args['sentences'][idx]}\nLABEL: {label}\nCERTAINTY: {score:.2f}\n")

tsk.close()
    

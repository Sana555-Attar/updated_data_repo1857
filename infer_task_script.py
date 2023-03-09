import joblib
from transformers import pipeline
from clearml import Model, Task, Logger

task = Task.init(
    project_name='detect_sarcasm',
    task_name='Sarcasm Inference',
    task_type='inference',
    reuse_last_task_id=False
)

# ab2f7493f0804244868221d76bd18a6a
transformer_model_path = Model(model_id="e270976c24e3420b964de12f774aaaf5").get_local_copy()  
sklearn_model_path = Model(model_id="fcdca53e3c554313b3035a64f53aa52e").get_local_copy()  

# transformer_pipeline = pipeline("text-classification", model="./my_awesome_model/checkpoint-best", device='cuda:0')
transformer_pipeline = pipeline("text-classification", model=transformer_model_path, device='cpu')
sklearn_pipeline = joblib.load(sklearn_model_path)


# sentence = "Hello World"

task.connect({"sentence" : "Hello World"})

def classify_transformer(sentence):
    sarcastic = transformer_pipeline(sentence)[0]
    return f"{sarcastic['label']}: {sarcastic['score']}"

def classify_sklearn(sentence):
    sarcastic = sklearn_pipeline.predict_proba([sentence])[0]
    if sarcastic[0] > sarcastic[1]:
        label = "NORMAL"
        score = sarcastic[0]
    else:
        label = "SARCASTIC"
        score = sarcastic[1]
    return f"{label}: {score}"

print("LogisticRegression Output:",classify_sklearn(sentence))
print("--------------------------")
print("DistilBERT Output:",classify_transformer(sentence))
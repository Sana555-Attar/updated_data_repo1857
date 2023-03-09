from clearml import Task
import tensorflow as tf
import pickle

#get the task
tsk = Task.init('sarcasm_detector','inferenceNN','inference')
task = Task.get_task('6a25d292a8ce4628ad3a77ccbca1e7b4')

#load the model
transformer_model_path = task.models.data['output'][0].get_local_copy()
model = tf.keras.models.load_model(transformer_model_path)
print(model.summary())

#load the tokenizer
tokenizer_path  = task.artifacts['local file'].get_local_copy()
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

args = {'sentence' : ["Coworkers At Bathroom Sink Locked In Tense Standoff Over Who Going To Wash Hands Longer","grandma jumps into buick for emergency birdseed run"]}
tsk.connect(args)
#tokenize input sentences
sequences = tokenizer.texts_to_sequences(args['sentence'])
print(f"Input Sentences :\n{args['sentence']}")
#pad sequences to max_len
padded =tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

res = model.predict(padded)
for idx, score in enumerate(res):
    score = score[0]
    if score < 0.5:
        label = 'NORMAL'
        score = 1 - score
    else:
        label = 'SARCASTIC'
    print(f"Commnet: {args['sentence'][idx]}\nLABEL: {label}\nCERTAINTY: {score:.2f}\n")

#print(res)
tsk.close()

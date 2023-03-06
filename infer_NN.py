from clearml import Task
import tensorflow as tf
import pickle

#get the task
tsk = Task.init('tf_sarcasm_detector','inferenceNN','inference')
task = Task.get_task('8df9a4c8e9dc43348c6c464858e6c8e4')

#load the model
transformer_model_path = task.models.data['output'][0].get_local_copy()
model = tf.keras.models.load_model(transformer_model_path)
print(model.summary())

#load the tokenizer
tokenizer_path  = task.artifacts['local file'].get_local_copy()
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

args = {'sentence' : ["Coworkers At Bathroom Sink Locked In Tense Standoff Over Who Going To Wash Hands Longer"]}
tsk.connect(args)
#tokenize input sentences
sequences = tokenizer.texts_to_sequences(args['sentence'])
print(args['sentence'])
#pad sequences to max_len
padded =tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

res = model.predict(padded)
print(res)
tsk.close()

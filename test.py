from numpy import loadtxt
from keras.models import model_from_json

dataset = loadtxt('pima-indians-diabetes.csv',delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
model.load_weights("model.h5")
print("loaded model from disk rajalakshmi")

prediction = model.predict_classes(x)
for i in range(5,10):
    print('%s => %d (original class: %d)' % (x[i].tolist(), prediction[i], y[i]))
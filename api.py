import logging
from flask import Flask, request, jsonify
from pickle import load as pload

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

api = Flask(__name__)

with open('car_evaluation_model.pkl', 'rb') as file:
    model = pload(file)
# vhigh, high, 2, more, big, med


@api.route('/api', methods=['POST'])
def predict_car_evaluation():
    if request.method == 'POST':

        # Get the data from the POST request
        data = request.get_json()

        # Extract the values from the received JSON
        buying = data.get('buying')
        maint = data.get('maint')
        doors = data.get('doors')
        persons = data.get('persons')
        lug_boot = data.get('lug_boot')
        safety = data.get('safety')

        # Make the prediction
        try:
            if (
                buying in model['predictor_categories']['buying'] and
                maint in model['predictor_categories']['maint'] and
                doors in model['predictor_categories']['doors'] and
                persons in model['predictor_categories']['persons'] and
                lug_boot in model['predictor_categories']['lug_boot'] and
                safety in model['predictor_categories']['safety']
            ):
                logger.info('All values are valid')
                prediction = model['model'].predict(
                    [[buying, maint, doors, persons, lug_boot, safety]]
                )
            else:
                logger.error('Invalid value in the request')
                return jsonify({'error': 'Invalid value'}), 400
        except Exception as e:
            logger.error('Invalid value in the request')
            return jsonify({'error': f'Error {e}'}), 400

        # Convert the prediction to a string
        class_predicted = model['target_classes'][int(prediction[0])]

        return jsonify({'prediction': class_predicted})


if __name__ == '__main__':
    api.run(debug=True)

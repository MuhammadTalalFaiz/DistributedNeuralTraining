from flask import Flask, render_template, request
from Train import train_model  # Import the train_model function

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    models = ['ResNet', 'DenseNet', 'VGG']  # Replace with your actual models
    databases = ['CIFAR-10', 'ImageNet', 'Database3']  # Replace with your actual databases

    selected_model = None
    selected_database = None
    training_progress = None  # Initialize training progress

    if request.method == 'POST':
        selected_model = request.form.get('model')
        selected_database = request.form.get('database')

        # Start training if model and dataset are selected
        if selected_model and selected_database:
            checkpoint_path, progress = train_model(selected_model, selected_database)
            training_progress = f"Training completed. Model saved at: {checkpoint_path}\n\n{progress}"

    return render_template('index.html', models=models, databases=databases, 
                           selected_model=selected_model, selected_database=selected_database,
                           training_progress=training_progress)

if __name__ == '__main__':
    app.run(debug=True)

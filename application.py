from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Initialize the Flask application
application = Flask(__name__)
app = application

# Load the pre-trained model and the StandardScaler
model = pickle.load(open("artifacts/ModelForPrediction.pkl", "rb"))
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))

# Load individual label encoders for categorical features
encoders = {}
categorical_columns = [
    'Gender', 'Smoking', 'Smoking_History', 'Radiotherapy_History', 'Thyroid_Function',
    'Physical_Examination', 'Adenopathy', 'Pathology', 'Focality', 'Tumor',
    'Lymph_Nodes', 'Cancer_Metastasis', 'Stage', 'Treatment_Response'
]

# Load each LabelEncoder for categorical columns
for col in categorical_columns:
    with open(f'artifacts/label_{col}.pkl', 'rb') as file:
        encoders[col] = pickle.load(file)

# Load the OrdinalEncoder for the 'Risk' column
risk_encoder = pickle.load(open('artifacts/label_Risk.pkl', 'rb'))

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for Single data point prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    result = "Res"

    if request.method == 'POST':
        try:
            # Print received form data for debugging
            print("Received Form Data:", request.form)

            # Fetching data from form input
            age = request.form.get("Age")
            gender = request.form.get("Gender")
            smoking = request.form.get("Smoking")
            hx_smoking = request.form.get("Hx_Smoking")
            hx_radiotherapy = request.form.get("Hx_Radiotherapy")
            thyroid_function = request.form.get("Thyroid_Function")
            physical_exam = request.form.get("Physical_Examination")
            adenopathy = request.form.get("Adenopathy")
            pathology = request.form.get("Pathology")
            focality = request.form.get("Focality")
            risk = request.form.get("Risk")
            tumor = request.form.get("T")
            lymph_nodes = request.form.get("N")
            cancer_metastasis = request.form.get("M")
            stage = request.form.get("Stage")
            treatment_response = request.form.get("Response")

            # Ensure that no values are None
            form_fields = {
                'Age': age, 'Gender': gender, 'Smoking': smoking, 'Smoking_History': hx_smoking,
                'Radiotherapy_History': hx_radiotherapy, 'Thyroid_Function': thyroid_function,
                'Physical_Examination': physical_exam, 'Adenopathy': adenopathy, 'Pathology': pathology,
                'Focality': focality, 'Risk': risk, 'Tumor': tumor, 'Lymph_Nodes': lymph_nodes,
                'Cancer_Metastasis': cancer_metastasis, 'Stage': stage, 'Treatment_Response': treatment_response
            }

            # Check for None values in form_fields
            for key, value in form_fields.items():
                if value is None:
                    print(f"Warning: Missing value for '{key}'")

            # Creating the input data dictionary
            data_dict = {key: value for key, value in form_fields.items()}

            # Convert to DataFrame
            input_df = pd.DataFrame([data_dict])

            # Print the created DataFrame for debugging
            print("Input DataFrame created successfully:")
            print(input_df)

            # Separate numerical and categorical features
            numerical_features = ['Age']
            categorical_features = [col for col in input_df.columns if col not in numerical_features]

            # Scale numerical features using the loaded StandardScaler
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])

            # Encode categorical features using the loaded LabelEncoders
            for column in categorical_features:
                if column == 'Risk':
                    # Use OrdinalEncoder for the 'Risk' column
                    input_df[column] = risk_encoder.transform(input_df[[column]])
                elif column in encoders:
                    # Check for missing values and replace with 'Unknown' before transforming
                    if input_df[column].isnull().any():
                        input_df[column].fillna('Unknown', inplace=True)
                    encoder = encoders[column]
                    input_df[column] = encoder.transform(input_df[column])
                else:
                    print(f"Warning: No encoder found for column '{column}' in encoders.")

            # Print the DataFrame after transformation
            print("DataFrame after scaling and encoding:")
            print(input_df)

            # Make prediction
            prediction = model.predict(input_df)

            # Output the result
            if prediction[0] == 1:
                result = 'The person is Thyroid Disease Positive'
            else:
                result = 'The person is Thyroid Disease Negative'

            input_data = data_dict
            return render_template('home.html', results=result, input_data=input_data)

        except Exception as e:
            print(f"An error occurred: {e}")
            return render_template('home.html', results="An error occurred during prediction.")

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

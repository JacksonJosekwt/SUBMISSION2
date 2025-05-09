from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import joblib
import os
import glob

app = Flask(__name__)

# --- Configuration ---
MODEL_DIR = 'saved_models'
STATIC_DIR = 'static'

# --- Global variables ---
# (Keep these as they are)
model = None
preprocessor = None
id_value_mappings = {}
field_descriptions = {}
feature_importance_table = [] 
field_order_preference = [] # This will be used to identify top features
default_suggestions = {}
available_zips = []
available_fips = []
numeric_model_features = []
categorical_model_features = []
expected_input_columns = []

TOP_N_MAIN_FEATURES = 6 # Define how many top features to show initially

def find_model_file(directory, pattern="*_final_model_v_part2.joblib"):
    search_pattern = os.path.join(directory, pattern)
    model_files = glob.glob(search_pattern)
    if not model_files:
        raise FileNotFoundError(f"No model file found: {pattern} in {directory}")
    return model_files[0]

def load_artifacts():
    global model, preprocessor, id_value_mappings, field_descriptions, \
           feature_importance_table, field_order_preference, default_suggestions, \
           available_zips, available_fips, numeric_model_features, categorical_model_features, \
           expected_input_columns

    print("Loading artifacts...")
    try:
        model_file_path = find_model_file(MODEL_DIR)
        model = joblib.load(model_file_path)
        print(f"Model loaded from {model_file_path}")

        preprocessor = joblib.load(os.path.join(MODEL_DIR, 'preprocessor_fitted_part2.joblib'))
        id_value_mappings = joblib.load(os.path.join(MODEL_DIR, 'id_value_mappings_final.joblib'))
        field_descriptions = joblib.load(os.path.join(MODEL_DIR, 'field_descriptions_final.joblib'))
        feature_importance_table = joblib.load(os.path.join(MODEL_DIR, 'feature_importance_table_data_final.joblib'))
        field_order_preference = joblib.load(os.path.join(MODEL_DIR, 'field_order_preference_final.joblib'))
        default_suggestions = joblib.load(os.path.join(MODEL_DIR, 'default_suggestions_final.joblib'))
        available_zips = joblib.load(os.path.join(MODEL_DIR, 'available_zips_clean_final.joblib'))
        available_fips = joblib.load(os.path.join(MODEL_DIR, 'available_fips_clean_final.joblib'))
        numeric_model_features = joblib.load(os.path.join(MODEL_DIR, 'numeric_features_for_preprocessor_final.joblib'))
        categorical_model_features = joblib.load(os.path.join(MODEL_DIR, 'categorical_features_for_preprocessor_final.joblib'))
        expected_input_columns = joblib.load(os.path.join(MODEL_DIR, 'expected_columns_before_preprocessing_final.joblib'))
        
        print("All artifacts loaded successfully.")
        print(f"Field order preference loaded (first 5): {field_order_preference[:5]}")

    except FileNotFoundError as e:
        print(f"ERROR: Could not load artifacts. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def _create_field_info(field_name):
    """Helper to create field_info dict for a single field."""
    field_info = {'name': field_name, 'label': field_name.replace('_', ' ').title()}
    field_info['description'] = field_descriptions.get(field_name.lower(), '')
    field_info['default'] = default_suggestions.get(field_name, '')

    if field_name in numeric_model_features:
        field_info['type'] = 'number'
        if any(s in field_name.lower() for s in ['year', 'count', 'number', 'num']) and not isinstance(field_info['default'], float):
            field_info['step'] = '1'
        else:
            field_info['step'] = 'any'
    elif field_name in categorical_model_features:
        field_info['type'] = 'select'
        id_map_key = field_name.lower()
        if id_map_key in id_value_mappings:
            field_info['options'] = id_value_mappings[id_map_key]
        elif field_name == 'regionidzip':
            field_info['options'] = {str(zip_val): str(zip_val) for zip_val in available_zips}
        elif field_name == 'fips':
            field_info['options'] = {str(fips_val): str(fips_val) for fips_val in available_fips}
        else:
            if field_info['default']:
                 field_info['options'] = {str(field_info['default']): str(field_info['default'])}
            else:
                 field_info['type'] = 'text'
    else:
        field_info['type'] = 'text'
    return field_info

def get_form_fields_structured():
    """Prepares structured lists of form fields (main and advanced)."""
    main_form_fields = []
    advanced_form_fields = []
    
    all_model_features = numeric_model_features + categorical_model_features
    
    # Use field_order_preference to determine the order and top features
    # Fallback if field_order_preference is not well-populated
    if not field_order_preference or len(field_order_preference) < TOP_N_MAIN_FEATURES :
        print(f"Warning: field_order_preference has < {TOP_N_MAIN_FEATURES} items or is empty. Using a mix of numeric/categorical for main.")
        # Simple fallback: take first few numeric and categorical
        main_field_names = numeric_model_features[:TOP_N_MAIN_FEATURES//2] + \
                             categorical_model_features[:TOP_N_MAIN_FEATURES - (TOP_N_MAIN_FEATURES//2)]
        # Ensure unique names if fallback is rough
        main_field_names = list(dict.fromkeys(main_field_names)) 
        if len(main_field_names) > TOP_N_MAIN_FEATURES:
            main_field_names = main_field_names[:TOP_N_MAIN_FEATURES]
            
        processed_main_names = set()
        for field_name in main_field_names:
            if field_name in all_model_features and field_name != 'saleprice':
                main_form_fields.append(_create_field_info(field_name))
                processed_main_names.add(field_name)

        for field_name in all_model_features:
            if field_name not in processed_main_names and field_name != 'saleprice':
                advanced_form_fields.append(_create_field_info(field_name))
                
    else: # Use field_order_preference
        top_field_names = [f for f in field_order_preference if f in all_model_features and f != 'saleprice'][:TOP_N_MAIN_FEATURES]
        
        processed_top_names = set()
        for field_name in top_field_names:
            main_form_fields.append(_create_field_info(field_name))
            processed_top_names.add(field_name)
            
        # Add remaining fields from field_order_preference to advanced
        for field_name in field_order_preference:
            if field_name not in processed_top_names and field_name in all_model_features and field_name != 'saleprice':
                advanced_form_fields.append(_create_field_info(field_name))
                processed_top_names.add(field_name) # Add to processed set to avoid duplication below

        # Add any other model features not yet included (e.g., if field_order_preference was incomplete)
        for field_name in all_model_features:
            if field_name not in processed_top_names and field_name != 'saleprice':
                advanced_form_fields.append(_create_field_info(field_name))

    print(f"Main form fields count: {len(main_form_fields)}")
    print(f"Advanced form fields count: {len(advanced_form_fields)}")
    return main_form_fields, advanced_form_fields


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    form_data_repop = {}
    top_feature_name = None
    top_feature_value = None
    top_feature_label = None

    if request.method == 'POST':
        form_data_repop = request.form.to_dict()
        try:
            input_data = {}
            # Construct input_data based on ALL expected_input_columns
            all_form_field_names_from_ui = set(field['name'] for field_list in get_form_fields_structured() for field in field_list)

            for col in expected_input_columns:
                if col not in all_form_field_names_from_ui: # If an expected col isn't even in the form list
                    print(f"Critical Warning: Column '{col}' expected by preprocessor is not defined in form fields. Using NaN.")
                    input_data[col] = np.nan
                    continue

                val = form_data_repop.get(col, default_suggestions.get(col, '')) # Get from form or default
                
                if col in numeric_model_features:
                    try:
                        input_data[col] = float(val) if val not in [None, ''] else np.nan
                    except ValueError:
                        input_data[col] = np.nan
                        print(f"Warning: Could not convert '{val}' to float for '{col}'. Using NaN.")
                elif col in categorical_model_features:
                    input_data[col] = str(val) if val not in [None, ''] else 'Unknown' # Or handle as per imputer
                else: # Column might be in expected_input_columns but not directly a model feature (will be dropped)
                    input_data[col] = str(val) if val not in [None, ''] else np.nan 
            
            input_df = pd.DataFrame([input_data], columns=expected_input_columns)
            
            print("\nInput DataFrame to preprocessor (head & dtypes):")
            print(input_df.head().to_string()) # to_string for better console output
            print(input_df.dtypes)

            input_processed = preprocessor.transform(input_df)
            pred_value = model.predict(input_processed)
            prediction = f"${pred_value[0]:,.2f}"

            if feature_importance_table:
                top_feature_info = feature_importance_table[0]
                top_feature_name = top_feature_info.get('feature')
                if top_feature_name:
                    top_feature_label = top_feature_name.replace('_', ' ').title()
                    top_feature_value = form_data_repop.get(top_feature_name, "N/A")
                    if top_feature_name.lower() in id_value_mappings:
                        try:
                            lookup_val = top_feature_value
                            if all(isinstance(k, int) for k in id_value_mappings[top_feature_name.lower()].keys()):
                                lookup_val = int(float(top_feature_value))
                            elif all(isinstance(k, float) for k in id_value_mappings[top_feature_name.lower()].keys()):
                                lookup_val = float(top_feature_value)
                            else:
                                lookup_val = str(top_feature_value)
                            top_feature_value = id_value_mappings[top_feature_name.lower()].get(lookup_val, top_feature_value)
                        except ValueError:
                             pass 
        except Exception as e:
            error_message = f"Error during prediction: {e}"
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()

    main_form_fields, advanced_form_fields = get_form_fields_structured()
    
    plots_for_template = {}
    plot_files = {
        "Aggregated Feature Importance": "feature_importance_aggregated_original_part2.png",
        "Processed Feature Importance": "feature_importance_processed_part2.png",
        "SHAP Bar Plot": "shap_summary_bar_part2.png",
        "SHAP Dot Plot": "shap_summary_dots_part2.png",
        "Model Performance": "model_performance_comparison_part2.png",
        "Predicted vs Actual (Best Model)": f"predicted_vs_actual_{model.__class__.__name__.lower().replace(' ', '_')}.png", # Assuming best model name is accessible or use a generic one
        "Residuals vs Predicted (Best Model)": f"residuals_vs_predicted_{model.__class__.__name__.lower().replace(' ', '_')}.png",
        "Residuals Distribution (Best Model)": f"residuals_distribution_{model.__class__.__name__.lower().replace(' ', '_')}.png"
    }
    # Attempt to get best model name from the model file if saved with a pattern
    try:
        best_model_filename = find_model_file(MODEL_DIR)
        best_model_base_name = os.path.basename(best_model_filename).replace('_final_model_v_part2.joblib', '')
        plot_files["Predicted vs Actual (Best Model)"] = f"predicted_vs_actual_{best_model_base_name}.png"
        plot_files["Residuals vs Predicted (Best Model)"] = f"residuals_vs_predicted_{best_model_base_name}.png"
        plot_files["Residuals Distribution (Best Model)"] = f"residuals_distribution_{best_model_base_name}.png"
    except Exception:
        print("Could not determine best model name for specific EDA plot filenames, using generic names.")


    for title, filename in plot_files.items():
        if os.path.exists(os.path.join(STATIC_DIR, 'plots', filename)):
            plots_for_template[title] = url_for('static', filename=f'plots/{filename}')
        else:
            print(f"Plot file not found: {os.path.join(STATIC_DIR, 'plots', filename)}")


    return render_template('index.html', 
                           main_form_fields=main_form_fields,         # <-- CHANGED
                           advanced_form_fields=advanced_form_fields, # <-- ADDED
                           prediction=prediction, 
                           error_message=error_message,
                           form_data=form_data_repop,
                           feature_importance_table=feature_importance_table,
                           plots=plots_for_template,
                           field_descriptions=field_descriptions,
                           top_feature_name=top_feature_name,
                           top_feature_value=top_feature_value,
                           top_feature_label=top_feature_label
                           )

if __name__ == '__main__':
    load_artifacts() 
    if model is None or preprocessor is None:
        print("CRITICAL ERROR: Model or Preprocessor not loaded.")
    app.run(debug=True)
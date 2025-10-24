import pandas as pd
import google.generativeai as genai
import os
import time

### 1. API AND MODEL CONFIGURATION ###
def run_nld_generation():
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        print("Gemini API Key configured successfully from environment variables.")
    except KeyError:
        print("ERROR: The GEMINI_API_KEY environment variable was not found.")
        print("Please set it or insert the key directly into the script in the API_KEY variable.")
        exit()

    MODEL_NAME = os.environ["LLM_MODEL_NAME"]
    MODEL_TEMPERATURE = float(os.environ.get("LLM_MODEL_TEMPERATURE", 0.0))

    INPUT_FILE = os.environ["CONSOLIDATED_NER_RESULTS"]
    OUTPUT_FILE = os.environ["CONSOLIDATED_NER_RESULTS_WITH_NLDS"]
    OUTPUT_FAILURE_FILE = os.environ["OUTPUT_FAILURE_FILE"]

    generation_config = genai.GenerationConfig(
        temperature=MODEL_TEMPERATURE,
    )

    def load_terms_and_labels_from_csv(filepath):
        """Loads terms and their labels from a CSV file."""
        if not os.path.exists(filepath):
            print(f"ERROR: The file '{filepath}' was not found.")
            return None
        try:
            df = pd.read_csv(filepath, encoding='utf-8', delimiter=',', header=0, usecols=['Readable_Term', 'Label'])
            print(f"Success! {len(df)} terms and labels loaded from '{filepath}'.")
            return df
        except Exception as e:
            print(f"ERROR reading the CSV file: {e}")
            return None


    system_instruction_correcao = "You are a data processing assistant specializing in correcting and standardizing technical terms from the geology domain."

    prompt_template_correcao = """"Your task is to correct and format the following technical term according to strict geological and petroleum domain standards. Follow these rules precisely:
    
    1. If the term is a concatenated phrase, separate the words (e.g., "carbonatemounds" -> "carbonate mounds").
    2. If the term contains an obvious typo, correct it.
    3. If the term is already correct and well-formatted, return it unchanged.
    4. If the term is nonsensical or unrecognizable return the exact string "UNKNOWN_TERM".
    
    Your response must contain ONLY the corrected term or the "UNKNOWN_TERM" flag.
    
    Term to be corrected:
    "{termo_bruto}"
    """

    system_instruction_definicao = "You are a senior geoscientist and ontology engineer. Your expertise is in oil and gas exploration geology, with a specific focus on the carbonate reservoirs of the Brazilian Pre-Salt."
    prompt_template_definicao = """Generate a concise and precise Natural Language Definition (NLD) for the provided term, using the assigned label as context for disambiguation.
    
    Mandatory Instructions:
    1. The definition must strictly follow the Aristotelian structure "X is a Y that Z". For example, "An amount of rock is a solid consolidated earth material that is constituted by an aggregate of particles made of mineral matter or material of biological origin"
    2. **Contextual Disambiguation:** You should use the `Label` to resolve any ambiguity in the term. For example, if the `Term to be defined` is "Paraná" and the assigned `Label` is "BACIA", you must define the Paraná Basin, not the river or the state.
    3. The definition should be technical yet clear, and a maximum of three sentences.
    4. Your response must contain only the generated NLD, without any extra text.
    
    Term to be defined: "{termo_corrigido}"
    Assigned Label: "{rotulo_ner}"
    """

    df_termos = load_terms_and_labels_from_csv(INPUT_FILE)

    if df_termos is not None:
        resultados = []
        termos_para_revisao = []

        model_correcao = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_correcao,
                                               generation_config=generation_config)
        model_definicao = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_definicao,
                                                generation_config=generation_config)

        total_termos = len(df_termos)
        for index, row in df_termos.iterrows():
            termo_bruto = row['Readable_Term']
            rotulo_ner = row['Label']

            # Translated log
            print(f"Processing term {index + 1}/{total_termos}: '{termo_bruto}'...")

            try:
                response_correcao = model_correcao.generate_content(
                    prompt_template_correcao.format(termo_bruto=termo_bruto))
                termo_corrigido = response_correcao.text.strip()

                if termo_corrigido == "UNKNOWN_TERM":
                    motivo = 'Not recognized by LLM' if termo_corrigido == "UNKNOWN_TERM" else 'Invalid response from correction LLM'
                    print(f"  -> Term '{termo_bruto}' invalid. Marked for manual review. Reason: {motivo}")
                    termos_para_revisao.append({'Term_Original': termo_bruto, 'Label': rotulo_ner, 'Reason': motivo, 'LLM_Response': termo_corrigido}) # Use English keys
                    time.sleep(1)
                    continue

                response_definicao = model_definicao.generate_content(
                    prompt_template_definicao.format(termo_corrigido=termo_corrigido, rotulo_ner=rotulo_ner))
                nld_gerada = response_definicao.text.strip()

                termo_corrigido_lower = termo_corrigido.lower()
                nld_lower = nld_gerada.lower()

                # Translated log
                print(f"  -> Definition generated successfully.")
                resultados.append(
                    {'Corrected_Term': termo_corrigido, 'NLD': nld_gerada, 'Original_Label': rotulo_ner}) # Use English keys

                time.sleep(1)

            except Exception as e:
                # Translated log
                print(f"  -> ERROR processing term '{termo_bruto}': {e}")
                termos_para_revisao.append({'Term_Original': termo_bruto, 'Label': rotulo_ner, 'Error': str(e)}) # Use English keys

        # Translated log
        print("\nProcessing complete. Saving results...")

        df_resultados = pd.DataFrame(resultados)
        df_resultados.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        # Translated log (using variable for filename)
        print(f"{len(df_resultados)} definitions saved to '{OUTPUT_FILE}'")

        if termos_para_revisao:
            df_revisao = pd.DataFrame(termos_para_revisao)
            df_revisao.to_csv(OUTPUT_FAILURE_FILE, index=False, encoding='utf-8-sig')
            # Translated log (using variable for filename)
            print(f"{len(df_revisao)} terms marked for manual review saved to '{OUTPUT_FAILURE_FILE}'")
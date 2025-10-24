from ner_term_extractor.ner_term_extractor_1_1 import run_ner_term_extraction
from term_aggregators import term_aggregator_for_ner_model_output_1_2
from nld_generator.nld_generator_1_3 import run_nld_generation
from term_categorizer.term_categorizer import run_term_categorization

def main():
    run_ner_term_extraction()
    term_aggregator_for_ner_model_output_1_2.run_term_aggregation()
    run_nld_generation()
    run_term_categorization()

if __name__ == "__main__":
    main()

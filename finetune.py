import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import shutil
import time
from huggingface_hub import HfApi, login
import zipfile
import tempfile
import gc
import math  # Added for log operations

# Set page configuration
st.set_page_config(
    page_title="Gemma Fine-tuning UI",
    page_icon="💎",
    layout="wide"
)

# App title and description
st.title("Gemma Fine-tuning UI 💎")
st.markdown("""
This application allows you to easily fine-tune Google's Gemma models on your own datasets.
Simply upload your data, configure the training parameters, and start fine-tuning!
""")

# Sidebar for authentication and model selection
with st.sidebar:
    st.header("Authentication")
    hf_token = st.text_input("Hugging Face Token (Required for Gemma)", type="password")
    
    if hf_token:
        try:
            login(token=hf_token)
            st.success("Token verified successfully!")
            token_valid = True
        except Exception as e:
            st.error(f"Token validation failed: {e}")
            token_valid = False
    else:
        st.warning("Please enter your Hugging Face token")
        token_valid = False

    st.header("Model Selection")
    model_options = [
        "google/gemma-2b",
        "google/gemma-2b-it",
        "google/gemma-7b", 
        "google/gemma-7b-it"
    ]
    selected_model = st.selectbox("Select Gemma Model", model_options)

# Function to format chat conversations for Gemma
def format_gemma_prompt(instruction, response=None):
    if response:
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{response}<end_of_turn>"
    else:
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

# Function to preprocess dataset
def preprocess_dataset(df, instruction_col, response_col=None):
    if response_col:
        # Instruction-response pairs
        df["text"] = df.apply(
            lambda row: format_gemma_prompt(row[instruction_col], row[response_col]), 
            axis=1
        )
    else:
        # Only instructions
        df["text"] = df[instruction_col].apply(format_gemma_prompt)
    
    return Dataset.from_pandas(df[["text"]])

# Function to tokenize dataset
def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )

# Function to load and prepare model
@st.cache_resource
def load_model(model_name, token):
    if not token:
        st.error("Please enter a valid Hugging Face token")
        return None, None
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        
        # Set padding token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Custom data collator that uses the tokenizer
def data_collator(features, tokenizer):
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for feature in features:
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        
        # Use input_ids as labels for causal language modeling
        labels = input_ids.clone()
        # Set padding tokens to -100 to ignore them in loss calculation
        labels[labels == tokenizer.pad_token_id] = -100
        
        batch["input_ids"].append(input_ids)
        batch["attention_mask"].append(attention_mask)
        batch["labels"].append(labels)
    
    # Convert to tensors
    batch = {k: torch.stack(v) for k, v in batch.items()}
    return batch

# Main training function
def train_model(model, tokenizer, dataset, validation_dataset, training_args):
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        data_collator=lambda features: data_collator(features, tokenizer)
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    
    # Return training metrics
    return trainer.state.log_history

# Data upload section
st.header("1. Upload Your Dataset")
data_format = st.radio(
    "Select data format:",
    ["CSV", "JSON/JSONL", "Text Files (TXT)"]
)

uploaded_file = st.file_uploader(
    f"Upload your {data_format} file",
    type=["csv", "json", "jsonl", "txt"] if data_format == "Text Files (TXT)" else data_format.lower()
)

# Dataset preview and column selection
if uploaded_file is not None:
    try:
        if data_format == "CSV":
            df = pd.read_csv(uploaded_file)
        elif data_format == "JSON/JSONL":
            if uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:  # jsonl
                df = pd.read_json(uploaded_file, lines=True)
        else:  # TXT
            # Read text file
            text_content = uploaded_file.read().decode('utf-8')
            # Split by lines and create a dataframe
            lines = text_content.split('\n')
            lines = [line for line in lines if line.strip()]  # Remove empty lines
            df = pd.DataFrame({'text': lines})
        
        st.write(f"Dataset preview ({len(df)} rows):")
        st.dataframe(df.head(5))
        
        # Column selection for CSV and JSON
        if data_format in ["CSV", "JSON/JSONL"]:
            columns = df.columns.tolist()
            
            st.subheader("Select columns for fine-tuning")
            is_instruction_response = st.checkbox(
                "Dataset contains instruction-response pairs", 
                value=True
            )
            
            if is_instruction_response:
                col1, col2 = st.columns(2)
                with col1:
                    instruction_col = st.selectbox("Instruction/prompt column", columns)
                with col2:
                    response_col = st.selectbox("Response/completion column", columns)
            else:
                instruction_col = st.selectbox("Text column", columns)
                response_col = None
        else:
            # For text files - assuming single column of text
            instruction_col = 'text'
            response_col = None
            is_instruction_response = False
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        df = None

# Training configuration section
st.header("2. Configure Fine-tuning")
with st.expander("Training Parameters", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        # Core parameters
        training_method = st.radio("Training Method", ["LoRA (recommended)", "Full Fine-tuning"])
        epochs = st.slider("Number of epochs", min_value=1, max_value=10, value=3)
        batch_size = st.slider("Batch size", min_value=1, max_value=32, value=4)
        
        # COMPLETELY REDESIGNED Learning Rate Section with Logarithmic Slider
        st.subheader("Learning Rate")

        # Define the range for the learning rate in log scale
        log_min_lr = -6  # Corresponds to 1e-6
        log_max_lr = -2  # Corresponds to 1e-2
        log_default_lr = -4  # Corresponds to 1e-4

        # Slider for learning rate in log scale
        log_lr = st.slider(
            "Learning Rate (log scale)",
            min_value=log_min_lr,
            max_value=log_max_lr,
            value=log_default_lr,
            step=1,
            format="1e%.0f"
        )

        # Convert the log scale value to the actual learning rate
        learning_rate = 10 ** log_lr

        # Display the chosen learning rate
        st.info(f"Selected Learning Rate: {learning_rate:.8f}")
    
    with col2:
        # Advanced parameters
        max_length = st.slider("Max sequence length", min_value=128, max_value=2048, value=512)
        validation_split = st.slider("Validation split", min_value=0.0, max_value=0.3, value=0.1)
        
        if training_method == "LoRA (recommended)":
            lora_rank = st.slider("LoRA rank (r)", min_value=1, max_value=64, value=8)
            lora_alpha = st.slider("LoRA alpha", min_value=1, max_value=64, value=16)
            lora_dropout = st.slider("LoRA dropout", min_value=0.0, max_value=0.5, value=0.05, step=0.05)
            
            # Specially configured for Gemma
            lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]
            st.info(f"Using target modules for Gemma: {', '.join(lora_target_modules)}")

# Start training section
st.header("3. Start Fine-tuning")
start_button = st.button(
    "Start Fine-tuning",
    disabled=not token_valid or not uploaded_file
)

if start_button and token_valid and uploaded_file:
    # Load model and tokenizer
    with st.spinner("Loading model and tokenizer..."):
        model, tokenizer = load_model(selected_model, hf_token)
        
    if model and tokenizer:
        # Preprocess dataset
        with st.spinner("Preprocessing dataset..."):
            try:
                # Create processed dataset
                processed_dataset = preprocess_dataset(df, instruction_col, response_col)
                
                # Split dataset
                if validation_split > 0:
                    train_val_dict = processed_dataset.train_test_split(
                        test_size=validation_split,
                        seed=42
                    )
                    train_dataset = train_val_dict["train"]
                    val_dataset = train_val_dict["test"]
                    st.info(f"Train set: {len(train_dataset)} samples | Validation set: {len(val_dataset)} samples")
                else:
                    train_dataset = processed_dataset
                    val_dataset = None
                    st.info(f"Train set: {len(train_dataset)} samples | No validation set")
                
                # Tokenize datasets
                train_tokenized = train_dataset.map(
                    lambda examples: tokenize_function(examples, tokenizer, max_length),
                    batched=True,
                    remove_columns=["text"]
                )
                
                val_tokenized = None
                if val_dataset:
                    val_tokenized = val_dataset.map(
                        lambda examples: tokenize_function(examples, tokenizer, max_length),
                        batched=True,
                        remove_columns=["text"]
                    )
                
                # Setup LoRA if selected
                if training_method == "LoRA (recommended)":
                    lora_config = LoraConfig(
                        r=lora_rank,
                        lora_alpha=lora_alpha,
                        target_modules=lora_target_modules,
                        lora_dropout=lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM"
                    )
                    model = get_peft_model(model, lora_config)
                    model.print_trainable_parameters()
                
                # Setup output directory
                output_dir = "fine_tuned_gemma_model"
                if os.path.exists(output_dir):
                    shutil.rmtree(output_dir)
                os.makedirs(output_dir)
                
                # Show the actual learning rate being used
                st.write(f"Using learning rate: {learning_rate:.8f}")
                
                # Training arguments based on model size and configuration
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    evaluation_strategy="epoch" if val_tokenized else "no",
                    save_strategy="epoch",
                    learning_rate=learning_rate,  # Use the configured learning rate
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    lr_scheduler_type="cosine",
                    logging_dir="./logs",
                    logging_steps=10,
                    report_to="tensorboard",
                    fp16=torch.cuda.is_available(),
                )
                
                # Create progress indicators
                progress_text = "Fine-tuning in progress. Please wait..."
                progress_bar = st.progress(0)
                status_container = st.empty()
                chart_container = st.container()
                
                # Train model with progress updates
                start_time = time.time()
                
                # Run the training in blocks
                def train_with_updates():
                    with chart_container:
                        col1, col2 = st.columns(2)
                        loss_chart_placeholder = col1.empty()
                        metrics_placeholder = col2.empty()
                    
                    logs = []
                    total_steps = epochs * (len(train_tokenized) // batch_size)
                    completed_steps = 0
                    
                    # Create and configure trainer
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        train_dataset=train_tokenized,
                        eval_dataset=val_tokenized,
                        tokenizer=tokenizer,
                        data_collator=lambda features: data_collator(features, tokenizer)
                    )
                    
                    # Patch the _maybe_log_save_evaluate method to capture progress
                    original_log_method = trainer._maybe_log_save_evaluate
                    
                    def patched_log_method(*args, **kwargs):
                        nonlocal completed_steps, logs
                        output = original_log_method(*args, **kwargs)
                        
                        # Update progress
                        if hasattr(trainer.state, "log_history") and trainer.state.log_history:
                            latest_log = trainer.state.log_history[-1]
                            logs.append(latest_log)
                            
                            # Extract training step and update progress
                            if "step" in latest_log:
                                completed_steps = latest_log["step"]
                                progress = min(completed_steps / total_steps, 1.0)
                                progress_bar.progress(progress)
                            
                            # If evaluation metrics are available
                            if "eval_loss" in latest_log:
                                status_container.info(f"Epoch {latest_log.get('epoch', 'N/A'):.2f} | "
                                                    f"Training Loss: {latest_log.get('loss', 'N/A'):.4f} | "
                                                    f"Validation Loss: {latest_log.get('eval_loss', 'N/A'):.4f}")
                            else:
                                status_container.info(f"Epoch {latest_log.get('epoch', 'N/A'):.2f} | "
                                                    f"Training Loss: {latest_log.get('loss', 'N/A'):.4f}")
                            
                            # Plot training progress
                            plot_training_progress(logs, loss_chart_placeholder)
                            
                            # Display other metrics
                            display_metrics(logs, metrics_placeholder)
                        
                        return output
                    
                    # Replace method with patched version
                    trainer._maybe_log_save_evaluate = patched_log_method
                    
                    # Train model
                    trainer.train()
                    
                    # Save model
                    trainer.save_model(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    
                    # Save LoRA config if using LoRA
                    if training_method == "LoRA (recommended)":
                        model.save_pretrained(output_dir)
                    
                    # Return logs for final reporting
                    return logs
                
                # Function to plot training progress
                def plot_training_progress(logs, placeholder):
                    if logs:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        
                        # Extract training loss
                        epochs = [log.get("epoch", idx) for idx, log in enumerate(logs) if "loss" in log]
                        train_loss = [log.get("loss") for log in logs if "loss" in log]
                        
                        # Extract validation loss if available
                        eval_epochs = [log.get("epoch") for log in logs if "eval_loss" in log]
                        eval_loss = [log.get("eval_loss") for log in logs if "eval_loss" in log]
                        
                        # Plot training loss
                        if train_loss:
                            ax.plot(epochs, train_loss, label="Training Loss", marker="o")
                        
                        # Plot validation loss
                        if eval_loss:
                            ax.plot(eval_epochs, eval_loss, label="Validation Loss", marker="x")
                        
                        ax.set_xlabel("Epoch")
                        ax.set_ylabel("Loss")
                        ax.set_title("Training Progress")
                        ax.legend()
                        ax.grid(True, linestyle="--", alpha=0.7)
                        
                        placeholder.pyplot(fig)
                
                # Function to display other metrics
                def display_metrics(logs, placeholder):
                    if logs:
                        latest = logs[-1]
                        metrics_text = "### Training Metrics\n\n"
                        
                        # Format metrics
                        for key, value in latest.items():
                            if isinstance(value, (int, float)):
                                metrics_text += f"**{key}**: {value:.4f}\n\n"
                            else:
                                metrics_text += f"**{key}**: {value}\n\n"
                        
                        placeholder.markdown(metrics_text)
                
                # Start the training process
                try:
                    training_logs = train_with_updates()
                    
                    # Training complete
                    elapsed_time = time.time() - start_time
                    st.success(f"Fine-tuning completed in {elapsed_time/60:.2f} minutes!")
                    
                    # Create download links
                    create_download_options(output_dir, training_method)
                    
                except Exception as e:
                    st.error(f"Error during training: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                
                # Clear some memory
                del model
                torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                st.error(f"Error during dataset preprocessing: {e}")
                import traceback
                st.code(traceback.format_exc())

# Function to create download options
def create_download_options(model_dir, training_method):
    st.header("4. Download Fine-tuned Model")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create a zip file of the entire model
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            with zipfile.ZipFile(tmp_file.name, 'w', zipfile.ZIP_DEFLATED) as archive:
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        archive.write(
                            os.path.join(root, file),
                            os.path.relpath(os.path.join(root, file), os.path.dirname(model_dir))
                        )
            
            with open(tmp_file.name, "rb") as f:
                st.download_button(
                    label="Download Complete Model",
                    data=f,
                    file_name="fine_tuned_gemma_model.zip",
                    mime="application/zip"
                )
    
    with col2:
        if training_method == "LoRA (recommended)":
            st.info("""
            **Using LoRA Adapter**
            
            You've fine-tuned using LoRA which is more efficient and compact. 
            The downloaded model contains adapter weights that can be applied to the original model.
            """)
        else:
            st.info("""
            **Using Full Fine-tuning**
            
            You've performed full fine-tuning of all parameters.
            The downloaded model is a complete copy of the fine-tuned model.
            """)
    
    # Model usage instructions
    st.subheader("How to use your fine-tuned model")
    
    code_lora = '''
    # Load fine-tuned LoRA model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    
    # Load the base model and tokenizer
    model_name = "google/gemma-2b"  # Use the same base model you selected
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Load your fine-tuned LoRA weights
    model = PeftModel.from_pretrained(model, "./fine_tuned_gemma_model")
    
    # Generate text
    inputs = tokenizer("user: What is machine learning?", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    '''
    
    code_full = '''
    # Load fully fine-tuned model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load your fine-tuned model and tokenizer
    model_path = "./fine_tuned_gemma_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Generate text
    inputs = tokenizer("user: What is machine learning?", return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    '''
    
    if training_method == "LoRA (recommended)":
        st.code(code_lora, language="python")
    else:
        st.code(code_full, language="python")

# Display additional information when no fine-tuning is in progress
if not start_button or not uploaded_file:
    st.header("Example Dataset Format")
    
    st.subheader("CSV Format Example")
    example_csv = pd.DataFrame({
        'instruction': [
            'Explain machine learning in simple terms',
            'What are the benefits of exercise?'
        ],
        'response': [
            'Machine learning is when computers learn from examples instead of being explicitly programmed. It\'s like teaching a child by showing examples rather than explaining detailed rules.',
            'Regular exercise improves cardiovascular health, builds muscle strength, reduces stress, improves sleep quality, and can help maintain a healthy weight.'
        ]
    })
    st.dataframe(example_csv)
    
    st.subheader("Tips for Better Fine-tuning")
    st.markdown("""
    1. **Quality Data**: Use high-quality, diverse, and well-formatted training data.
    2. **Proper Formatting**: Format your prompts consistently using the Gemma chat template.
    3. **Hyperparameter Tuning**: Start with recommended parameters and experiment from there.
    4. **Model Size**: Choose the model size based on your task complexity and available resources.
    5. **Validation Set**: Always include a validation set to monitor overfitting.
    """)

# Footer
st.markdown("---")
st.markdown("""
**Gemma Fine-tuning UI** | Created for the Gemma ecosystem
""")

# Add CSS styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
    }
    .stProgress .st-bd {
        height: 20px;
    }
    h1, h2, h3 {
        color: #357736;
    }
    .stAlert {
        background-color: #f1f9f2;
    }
</style>
""", unsafe_allow_html=True)
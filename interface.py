import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

def update_model_description():
    """Atualiza o texto de descrição do modelo com base na seleção."""
    selected_model = model_selection_var.get()
    if selected_model == "MLP":
        description = "+ Preciso para Introvertido e Extrovertido"
        model_description_label.config(text=description, foreground="#0056b3")
    elif selected_model == "SVM":
        description = "+ Preciso para Ambivertido"
        model_description_label.config(text=description, foreground="#0056b3")

def predict_personality():
    """Coleta os dados do formulário, faz a predição e exibe o resultado."""
    try:
        time_spent_alone = int(time_spent_alone_var.get())
        stage_fear = 1 if stage_fear_var.get() == "Yes" else 0
        social_event_attendance = int(social_event_attendance_var.get())
        going_outside = int(going_outside_var.get())
        drained_after_socializing = 1 if drained_after_socializing_var.get() == "Yes" else 0
        friends_circle_size = int(friends_circle_size_var.get())
        post_frequency = int(post_frequency_var.get())

        df_input = pd.DataFrame([[
            time_spent_alone,
            stage_fear,
            social_event_attendance,
            going_outside,
            drained_after_socializing,
            friends_circle_size,
            post_frequency
        ]], columns=[
            'Time_spent_Alone',
            'Stage_fear',
            'Social_event_attendance',
            'Going_outside',
            'Drained_after_socializing',
            'Friends_circle_size',
            'Post_frequency'
        ])
        print("Dados de entrada para predição:\n", df_input) 

        normalizador_carregado = joblib.load('normalizador_scaler.pkl')
        df_normalizado = normalizador_carregado.transform(df_input)
        
        selected_model = model_selection_var.get()
        prediction = None

        if selected_model == "MLP":
            model = tf.keras.models.load_model('model.keras')
            prediction = model.predict(df_normalizado)
            prob_extrovert = prediction[0][0]
            print(f"Predição MLP bruta (probabilidade de Extrovertido): {prob_extrovert}")
        elif selected_model == "SVM":
            loaded_svm_model = joblib.load('rf_model.pkl')
            prediction_proba = loaded_svm_model.predict_proba(df_normalizado)
            prob_extrovert = prediction_proba[0][1]
            print(f"Predição SVM (ou RF) bruta (probabilidade de Extrovertido): {prob_extrovert}")
        else:
            messagebox.showerror("Erro de Seleção", "Por favor, selecione um modelo (MLP ou SVM/RF).")
            return

        confianca = f'{prob_extrovert:.2f}'
        if prob_extrovert >= 0.7: 
            predicted_personality = f"Extrovertido"
            detailed_msg = "Você demonstra fortes características de um extrovertido! Adora interagir, se sente energizado(a) na presença de outras pessoas e é bastante ativo(a) socialmente."
            color = "#28a745" 
        elif 0.5 < prob_extrovert < 0.7: 
            predicted_personality = f"Tendência a Extrovertido"
            detailed_msg = "Você possui um perfil ambivertido, com uma leve tendência a ser extrovertido. Gosta de socializar, mas também valoriza seus momentos de introspecção."
            color = "#ffc107" 
        elif 0.3 <= prob_extrovert <= 0.5: 
            predicted_personality = f"Tendência a Introvertido"
            detailed_msg = "Seu perfil é ambivertido, com uma inclinação para a introversão. Aprecia a tranquilidade e a reflexão, mas não se isola e pode gostar de interações sociais mais focadas."
            color = "#ffc107" 
        else: 
            predicted_personality = f"Introvertido"
            detailed_msg = "Você apresenta fortes traços de uma personalidade introvertida! Prefere a tranquilidade, recarrega as energias sozinho(a) e tende a ser mais reservado(a) em grupos grandes."
            color = "#dc3545" 

        result_label.config(text=f"Personalidade Prevista: {predicted_personality}\nPrevisão (0-1): {confianca}", foreground=color)
        detailed_message_label.config(text=detailed_msg, foreground="#343a40")

    except ValueError:
        messagebox.showerror("Erro de Entrada", "Por favor, insira valores válidos para todos os campos numéricos.")
    except FileNotFoundError as e:
        messagebox.showerror("Erro de Arquivo", f"Arquivo necessário não encontrado: {e}. Certifique-se de que 'normalizador_scaler.pkl', 'model.keras' e 'rf_model.pkl' estão no mesmo diretório.")
    except Exception as e:
        messagebox.showerror("Erro Inesperado", f"Ocorreu um erro: {e}")

def reset_inputs():
    """Redefine todos os campos do formulário para seus valores padrão."""
    time_spent_alone_var.set("0")
    stage_fear_var.set("No")
    social_event_attendance_var.set("0")
    going_outside_var.set("0")
    drained_after_socializing_var.set("No")
    friends_circle_size_var.set("0")
    post_frequency_var.set("0")
    model_selection_var.set("SVM") 
    result_label.config(text="Previsão: Aguardando entrada...", foreground="black") 
    detailed_message_label.config(text="", foreground="black") 
    update_model_description() 

root = tk.Tk()
root.title("Analisador de Personalidade")

window_width = 550
window_height = 750 
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
center_x = int(screen_width/2 - window_width / 2)
center_y = int(screen_height/2 - (window_height / 2 + 10))
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
root.resizable(False, False)


style = ttk.Style(root)
style.theme_use("clam")

style.configure("TLabel", font=("Helvetica", 10))
style.configure("TButton", font=("Helvetica", 10, "bold"), padding=8)
style.configure("TLabelFrame.Label", font=("Helvetica", 12, "bold"))
style.configure("Header.TLabel", font=("Helvetica", 16, "bold"), foreground="#0056b3")
style.configure("ModelDesc.TLabel", font=("Helvetica", 9, "italic"), foreground="#343a40")

title_label = ttk.Label(root, text="Analisador de Personalidade", style="Header.TLabel")
title_label.pack(pady=(15, 10))

model_selection_frame = ttk.LabelFrame(root, text="Seleção do Modelo", padding="15")
model_selection_frame.pack(padx=20, pady=5, fill="x")

model_selection_var = tk.StringVar(value="SVM") 
model_selection_var.trace_add("write", lambda *args: update_model_description()) 

ttk.Radiobutton(model_selection_frame, text="Rede Neural (MLP)", variable=model_selection_var, value="MLP").pack(side="left", padx=10)
ttk.Radiobutton(model_selection_frame, text="Máquina de Vetor de Suporte (SVM)", variable=model_selection_var, value="SVM").pack(side="left", padx=10)

model_description_label = ttk.Label(model_selection_frame, text="", wraplength=100, justify="center", style="ModelDesc.TLabel")
model_description_label.pack(pady=(10, 0))

form_frame = ttk.LabelFrame(root, text="Informações Pessoais", padding="20")
form_frame.pack(padx=20, pady=10, fill="x")

for i in range(7):
    form_frame.grid_rowconfigure(i, weight=1)
form_frame.grid_columnconfigure(0, weight=1)
form_frame.grid_columnconfigure(1, weight=1)

ttk.Label(form_frame, text="Horas sozinho(a) diariamente (0-11):").grid(row=0, column=0, sticky="w", pady=5)
time_spent_alone_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=11, textvariable=time_spent_alone_var, wrap=True, width=5).grid(row=0, column=1, sticky="w", padx=5)

ttk.Label(form_frame, text="Sente medo de palco?").grid(row=1, column=0, sticky="w", pady=5)
stage_fear_var = tk.StringVar(value="No")
ttk.Radiobutton(form_frame, text="Sim", variable=stage_fear_var, value="Yes").grid(row=1, column=1, sticky="w")
ttk.Radiobutton(form_frame, text="Não", variable=stage_fear_var, value="No").grid(row=1, column=1, sticky="e")

ttk.Label(form_frame, text="Frequência de eventos sociais (0-10):").grid(row=2, column=0, sticky="w", pady=5)
social_event_attendance_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=10, textvariable=social_event_attendance_var, wrap=True, width=5).grid(row=2, column=1, sticky="w", padx=5)

ttk.Label(form_frame, text="Frequência de sair de casa (0-7):").grid(row=3, column=0, sticky="w", pady=5)
going_outside_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=7, textvariable=going_outside_var, wrap=True, width=5).grid(row=3, column=1, sticky="w", padx=5)

ttk.Label(form_frame, text="Sente-se esgotado(a) após socializar?").grid(row=4, column=0, sticky="w", pady=5)
drained_after_socializing_var = tk.StringVar(value="No")
ttk.Radiobutton(form_frame, text="Sim", variable=drained_after_socializing_var, value="Yes").grid(row=4, column=1, sticky="w")
ttk.Radiobutton(form_frame, text="Não", variable=drained_after_socializing_var, value="No").grid(row=4, column=1, sticky="e")

ttk.Label(form_frame, text="Tamanho do círculo de amigos (0-15):").grid(row=5, column=0, sticky="w", pady=5)
friends_circle_size_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=15, textvariable=friends_circle_size_var, wrap=True, width=5).grid(row=5, column=1, sticky="w", padx=5)

ttk.Label(form_frame, text="Frequência de posts em redes sociais (0-10):").grid(row=6, column=0, sticky="w", pady=5)
post_frequency_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=10, textvariable=post_frequency_var, wrap=True, width=5).grid(row=6, column=1, sticky="w", padx=5)

button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

analyze_button = ttk.Button(button_frame, text="Analisar Personalidade", command=predict_personality)
analyze_button.grid(row=0, column=0, padx=10)

reset_button = ttk.Button(button_frame, text="Redefinir Campos", command=reset_inputs)
reset_button.grid(row=0, column=1, padx=10)

result_frame = ttk.LabelFrame(root, text="Resultado da Análise", padding="15")
result_frame.pack(padx=20, pady=10, fill="x")

result_label = ttk.Label(result_frame, text="Previsão: Aguardando entrada...", font=("Helvetica", 14, "bold"))
result_label.pack(pady=(5, 10))

detailed_message_label = ttk.Label(result_frame, text="", wraplength=450, justify="center", font=("Helvetica", 10, "italic"))
detailed_message_label.pack(pady=(0, 5))

update_model_description()

root.mainloop()
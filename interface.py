import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import tensorflow as tf
import numpy as np

def predict_personality():
    """Coleta os dados do formulário, faz a predição e exibe o resultado."""
    try:
        # Coletar os dados do formulário
        time_spent_alone = int(time_spent_alone_var.get())
        stage_fear = 1 if stage_fear_var.get() == "Yes" else 0
        social_event_attendance = int(social_event_attendance_var.get())
        going_outside = int(going_outside_var.get())
        drained_after_socializing = 1 if drained_after_socializing_var.get() == "Yes" else 0
        friends_circle_size = int(friends_circle_size_var.get())
        post_frequency = int(post_frequency_var.get())

        # Criar o DataFrame de entrada
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

        # Carregar o normalizador e normalizar os dados
        normalizador_carregado = joblib.load('normalizador_scaler.pkl')
        df_normalizado = normalizador_carregado.transform(df_input)
        df_normalizado = pd.DataFrame(df_normalizado, columns=df_input.columns)

        # Carregar o modelo e fazer a predição
        model = tf.keras.models.load_model('model.keras')
        prediction = model.predict(df_normalizado)

        # Interpretar a predição
        if prediction[0][0] > 0.5:
            predicted_personality = f"Extrovertido {prediction[0][0]:.2f}%"
            detailed_msg = "Você demonstra características de um extrovertido! Adora interagir, se sente energizado(a) na presença de outras pessoas e é bastante ativo(a) socialmente."
        else:
            predicted_personality = f"Introvertido {1 - prediction[0][0]:.2f}%"
            detailed_msg = "Você apresenta traços de uma personalidade introvertida! Prefere a tranquilidade, recarrega as energias sozinho(a) e pode ser mais reservado(a) em grupos grandes."

        # Exibir o resultado
        result_label.config(text=f"Personalidade Prevista: {predicted_personality}")
        detailed_message_label.config(text=detailed_msg)

    except ValueError:
        messagebox.showerror("Erro de Entrada", "Por favor, insira valores válidos para todos os campos numéricos.")
    except FileNotFoundError as e:
        messagebox.showerror("Erro de Arquivo", f"Arquivo necessário não encontrado: {e}. Certifique-se de que 'normalizador_scaler.pkl' e 'model.keras' estão no mesmo diretório.")
    except Exception as e:
        messagebox.showerror("Erro Inesperado", f"Ocorreu um erro: {e}")

# Configuração da janela principal
root = tk.Tk()
root.title("Analisador de Personalidade")
root.geometry("500x600")
root.resizable(False, False)

# --- Formulário de Entrada ---
form_frame = ttk.LabelFrame(root, text="Informações Pessoais", padding="20")
form_frame.pack(padx=20, pady=20, fill="both", expand=True)

# Dicionário para armazenar as variáveis de controle do Tkinter
entries = {}

# Time_spent_Alone
ttk.Label(form_frame, text="Horas passadas sozinho(a) diariamente (0-11):").grid(row=0, column=0, sticky="w", pady=5)
time_spent_alone_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=11, textvariable=time_spent_alone_var, wrap=True).grid(row=0, column=1, sticky="ew", pady=5)
entries['Time_spent_Alone'] = time_spent_alone_var

# Stage_fear
ttk.Label(form_frame, text="Sente medo de palco?").grid(row=1, column=0, sticky="w", pady=5)
stage_fear_var = tk.StringVar(value="No")
ttk.Radiobutton(form_frame, text="Sim", variable=stage_fear_var, value="Yes").grid(row=1, column=1, sticky="w")
ttk.Radiobutton(form_frame, text="Não", variable=stage_fear_var, value="No").grid(row=1, column=1, sticky="e")
entries['Stage_fear'] = stage_fear_var

# Social_event_attendance
ttk.Label(form_frame, text="Frequência de eventos sociais (0-10):").grid(row=2, column=0, sticky="w", pady=5)
social_event_attendance_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=10, textvariable=social_event_attendance_var, wrap=True).grid(row=2, column=1, sticky="ew", pady=5)
entries['Social_event_attendance'] = social_event_attendance_var

# Going_outside
ttk.Label(form_frame, text="Frequência de sair de casa (0-7):").grid(row=3, column=0, sticky="w", pady=5)
going_outside_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=7, textvariable=going_outside_var, wrap=True).grid(row=3, column=1, sticky="ew", pady=5)
entries['Going_outside'] = going_outside_var

# Drained_after_socializing
ttk.Label(form_frame, text="Sente-se esgotado(a) após socializar?").grid(row=4, column=0, sticky="w", pady=5)
drained_after_socializing_var = tk.StringVar(value="No")
ttk.Radiobutton(form_frame, text="Sim", variable=drained_after_socializing_var, value="Yes").grid(row=4, column=1, sticky="w")
ttk.Radiobutton(form_frame, text="Não", variable=drained_after_socializing_var, value="No").grid(row=4, column=1, sticky="e")
entries['Drained_after_socializing'] = drained_after_socializing_var

# Friends_circle_size
ttk.Label(form_frame, text="Tamanho do círculo de amigos próximos (0-15):").grid(row=5, column=0, sticky="w", pady=5)
friends_circle_size_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=15, textvariable=friends_circle_size_var, wrap=True).grid(row=5, column=1, sticky="ew", pady=5)
entries['Friends_circle_size'] = friends_circle_size_var

# Post_frequency
ttk.Label(form_frame, text="Frequência de posts em redes sociais (0-10):").grid(row=6, column=0, sticky="w", pady=5)
post_frequency_var = tk.StringVar(value="0")
ttk.Spinbox(form_frame, from_=0, to=10, textvariable=post_frequency_var, wrap=True).grid(row=6, column=1, sticky="ew", pady=5)
entries['Post_frequency'] = post_frequency_var

# Botão de Análise
analyze_button = ttk.Button(root, text="Analisar", command=predict_personality)
analyze_button.pack(pady=10)

# --- Seção de Resultados ---
result_frame = ttk.LabelFrame(root, text="Resultado da Análise", padding="10")
result_frame.pack(padx=20, pady=10, fill="both", expand=True)

result_label = ttk.Label(result_frame, text="Previsão: Aguardando entrada...", font=("Helvetica", 14, "bold"))
result_label.pack(pady=5)

detailed_message_label = ttk.Label(result_frame, text="", wraplength=400, justify="center", font=("Helvetica", 10))
detailed_message_label.pack(pady=5)

root.mainloop()
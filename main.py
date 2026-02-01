import os
import subprocess
from pathlib import Path
from faster_whisper import WhisperModel

# 1. Configuração do Modelo
model = WhisperModel("medium", device="cpu", compute_type="int8", cpu_threads=16)

pasta_input = Path("input")
pasta_output = Path("output")
pasta_output.mkdir(exist_ok=True)

extensoes = ("*.mp4", "*.mp3", "*.mkv", "*.wav", "*.m4a")
arquivos_para_processar = []
for ext in extensoes:
    arquivos_para_processar.extend(pasta_input.glob(ext))
    arquivos_para_processar.extend(pasta_input.glob(ext.upper()))

if not arquivos_para_processar:
    print(f"Nenhum arquivo encontrado em '{pasta_input}'.")
else:
    for caminho_arquivo in arquivos_para_processar:
        nome_base = caminho_arquivo.stem
        pasta_destino_final = pasta_output / nome_base
        
        if (pasta_destino_final / f"{nome_base}.txt").exists():
            print(f"Pulei: {nome_base}")
            continue

        print(f"\n--- Processando: {caminho_arquivo.name} ---")
        pasta_destino_final.mkdir(parents=True, exist_ok=True)
        pasta_recortes = pasta_destino_final / "audios"
        pasta_recortes.mkdir(exist_ok=True)
        nome_saida_txt = pasta_destino_final / f"{nome_base}.txt"

        # MELHORIA DE PRECISÃO: vad_filter ajuda a detectar melhor onde a fala começa/para
        segments, info = model.transcribe(
            str(caminho_arquivo), 
            beam_size=5, 
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500), # Ajusta sensibilidade do silêncio
            word_timestamps=True # Aumenta a precisão dos tempos internos
        )
        
        with open(nome_saida_txt, "w", encoding="utf-8") as f:
            for i, segment in enumerate(segments, start=1):
                num_frase = f"{i:03d}"
                caminho_audio = pasta_recortes / f"{num_frase}.mp3"
                
                # MARGEM DE SEGURANÇA: Adiciona 0.2s no fim para não cortar a fala bruscamente
                inicio = segment.start
                fim = segment.end + 0.20 
                duracao = fim - inicio

                if duracao < 0.5: continue

                # COMANDO FFMPEG REFINADO:
                # -accurate_seek e colocando o -ss DEPOIS do -i ajuda em alguns arquivos a serem mais precisos
                subprocess.run([
                    "ffmpeg", "-y", 
                    "-ss", str(inicio), 
                    "-t", str(duracao),
                    "-i", str(caminho_arquivo), 
                    "-q:a", "2", 
                    "-vn", 
                    str(caminho_audio)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                f.write(f"Frase {num_frase} | {segment.text.strip()}\n")
                print(f"[{num_frase}] {segment.text.strip()[:40]}...")
        
        print(f"✓ Concluído: {nome_base}")

print("\n--- Processo finalizado! ---")
#!/usr/bin/env bash
set -euo pipefail

cd /dpc-zhouy/zhouy/MindSpeed-LLM/
source /dpc-zhouy/zhouy/miniconda3/bin/activate
conda activate py311
source /dpc-zhouy/usr/local/Ascend/ascend-toolkit/set_env.sh

# Adjust these paths.
INPUT_ROOT="/dpc-zhouy/zhouy/post-training-data/Llama-Nemotron-Post-Training-Dataset/SFT"
CONVERTED_ROOT="/dpc-zhouy/zhouy/post-training-data/nemotron_sft_jsonl"
OUTPUT_ROOT="/dpc-zhouy/zhouy/instructft_data/nemotron_bin_4K"
TOKENIZER="/dpc-zhouy/zhouy/ckpts/Qwen3-8B"

WORKERS=32
SEQ_LENGTH=4096

mkdir -p "$CONVERTED_ROOT" "$OUTPUT_ROOT"

while IFS= read -r -d '' INPUT_FILE; do
    RELATIVE_PATH="${INPUT_FILE#"$INPUT_ROOT"/}"
    STEM="${RELATIVE_PATH%.jsonl}"

    CONVERTED_FILE="${CONVERTED_ROOT}/${STEM}.jsonl"
    OUTPUT_PREFIX="${OUTPUT_ROOT}/${STEM}"
    LOG_FILE="${OUTPUT_PREFIX}.preprocess.log"

    mkdir -p "$(dirname "$CONVERTED_FILE")"
    mkdir -p "$(dirname "$OUTPUT_PREFIX")"

    echo "Converting: $INPUT_FILE"

    python - "$INPUT_FILE" "$CONVERTED_FILE" <<'PY'
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

if source.resolve() == destination.resolve():
    raise ValueError("Input and output must be different files")

temporary = destination.with_name(
    destination.name + f".tmp.{os.getpid()}"
)

role_map = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
}

count = 0

try:
    with source.open(encoding="utf-8") as fin, \
         temporary.open("w", encoding="utf-8") as fout:

        for line_number, line in enumerate(fin, 1):
            if not line.strip():
                continue

            row = json.loads(line)
            messages = row.get("input")
            response = row.get("output")

            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"{source}:{line_number}: input must be a nonempty list"
                )

            if not isinstance(response, str) or not response.strip():
                raise ValueError(
                    f"{source}:{line_number}: missing or empty output"
                )

            system = row.get("system_prompt") or ""
            if not isinstance(system, str):
                raise ValueError(
                    f"{source}:{line_number}: system_prompt must be text"
                )

            conversations = []

            for index, message in enumerate(messages):
                role = message.get("role")
                content = message.get("content")

                if role not in role_map or not isinstance(content, str):
                    raise ValueError(
                        f"{source}:{line_number}: unsupported message "
                        f"role/content: {role!r}"
                    )

                # Resolve an existing leading system message explicitly.
                if role == "system":
                    if index != 0:
                        raise ValueError(
                            f"{source}:{line_number}: non-leading system message"
                        )
                    if content and system and content != system:
                        system = content + "\n\n" + system
                    elif content:
                        system = content
                    continue

                conversations.append({
                    "from": role_map[role],
                    "value": content,
                })

            conversations.append({
                "from": "gpt",
                "value": response,
            })

            # The selected ShareGPT handler expects user/assistant alternation.
            for index, message in enumerate(conversations):
                expected = "human" if index % 2 == 0 else "gpt"
                if message["from"] != expected:
                    raise ValueError(
                        f"{source}:{line_number}: expected {expected} "
                        f"at conversation position {index}"
                    )

            if len(conversations) % 2:
                raise ValueError(
                    f"{source}:{line_number}: incomplete conversation"
                )

            result = {
                "conversations": conversations,
                "system": system,
            }

            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        raise ValueError(f"No examples found in {source}")

    temporary.replace(destination)

except Exception:
    temporary.unlink(missing_ok=True)
    raise

print(f"Converted {count} examples -> {destination}")
PY

    echo "Preprocessing: $CONVERTED_FILE"
    echo "Output prefix: $OUTPUT_PREFIX"

    python preprocess_data.py \
        --input "$CONVERTED_FILE" \
        --output-prefix "$OUTPUT_PREFIX" \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path "$TOKENIZER" \
        --handler-name SharegptStyleInstructionHandler \
        --prompt-type qwen3 \
        --seq-length "$SEQ_LENGTH" \
        --workers "$WORKERS" \
        --log-interval 100000 \
        --cache-dir /dpc-zhouy/cache 

done < <(find "$INPUT_ROOT" -type f -name '*.jsonl' -print0)

echo "Finished. Output directory: $OUTPUT_ROOT"


# INFO:__main__:data example:{'system': [''], 'prompt': [{'content': 'aluminium plaNT', 'role': 'user'}], 'response': [{'content': 'Here\'s a comprehensive overview of an aluminium plant, covering various aspects from its definition to operational details. \n\n### Definition and Purpose\n\n- **Definition**: An aluminium plant is an industrial facility where aluminium is produced through a series of complex processes. Aluminium, being a highly reactive metal, is not found freely in nature and must be extracted from its ores, primarily bauxite.\n- **Purpose**: The primary purpose of an aluminium plant is to extract aluminium from bauxite through refining and smelting processes, producing pure aluminium metal that can be further processed into various forms (e.g., ingots, sheets, foil) for use in a wide range of industries, including aerospace, automotive, construction, and packaging.\n\n### Key Processes in an Aluminium Plant\n\n1. **Bauxite Mining and Refining**:\n   - **Mining**: Bauxite, the primary ore of aluminium, is mined from open-pit or underground deposits.\n   - **Refining (Bayer Process)**: Bauxite is refined into alumina (aluminium oxide) through the Bayer process, which involves grinding, digestion in sodium hydroxide, clarification, and precipitation.\n\n2. **Smelting (Electrolysis)**:\n   - **Process**: Alumina is then smelted into pure aluminium metal through electrolysis, known as the Hall-Héroult process. This involves dissolving alumina in a bath of molten cryolite and then applying an electric current to reduce the alumina to aluminium.\n   - **Machinery**: The smelting process uses large electrolytic cells or pots, which are lined with carbon and connected to a power supply. \n   - **Outcome**: The result is molten aluminium, which is then cast into various shapes.\n\n3. **Casting**:\n   - **Process**: The molten aluminium from the smelting process is cast intodifferent forms to facilitate further processing or direct use. Common forms include ingots, billets, and slabs.\n   - **Machinery**: Casting involves the use of moulds, crucibles, and casting machines to shape the molten aluminium into the desired forms.\n   - **Outcome**: These cast forms can be sold as is or further processed.\n\n4. **Rolling and Extrusion (Secondary Processing)**:\n   - **Rolling**: For producing sheets, plates, and foils, the cast slabs are rolled to the desired thickness using heavy rolling mills.\n   - **Extrusion**: For creating shapes like rods, bars, and complex profiles, billets are heated and pushed through a die under high pressure using hydraulic presses.\n   - **Outcome**: Finished or semi-finished aluminium products ready for various applications.\n\n### Environmental and Health Considerations\n\n- **Environmental Impact**: Aluminium production is energy-intensive and can have significant environmental impacts, including greenhouse gas emissions, fluoride pollution (from the smelting process), and bauxite mining\'s habitat destruction. Modern plants are implementing more sustainable practices and technologies to mitigate these effects.\n- **Health and Safety**: Workers in aluminium plants are exposed to risks such as lung diseases from inhaling dust and fumes, electrical hazards, and physical injuries. Strict safety protocols and personal protective equipment (PPE) are crucial.\n\n### Technological Innovations and Sustainability Efforts\n\n- **Renewable Energy Integration**: Many aluminium plants are transitioning to renewable energy sources to reduce their carbon footprint.\n- **Recycling**: Increasing emphasison recycling aluminium to conserve energy (up to 95% less than primary production) and reduce waste.\n- **Advanced Smelting Technologies**: Innovations like more efficient electrolysis cells and inert anode technologies aim to reduce emissions and energy consumption.\n\nIf you are referring to a different kind of "aluminium plant" (e.g., a decorative plant with aluminium-like features), please provide more context for a tailored response.\n\n### Conclusion\n\nAn aluminium plant plays a critical role in supplying the world with a versatile metal essential for modern life. As the global demand for aluminium continues to grow, the industry is under increasing pressure to adopt more sustainable and environmentally friendly practices.', 'role': 'assistant'}], 'tools': ['']}
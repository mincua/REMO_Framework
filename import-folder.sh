#!/bin/bash

folder_path=""
web_service_url="http://127.0.0.1:3000/add_message"
speaker="SecondBrain"

cd "$folder_path" || exit 1

while IFS= read -r -d '' file_path; do
    file_content=$(cat "$file_path")
    file_created_at=$(stat -f %B "$file_path")

    # Properly concatenate the filename and content with JSON encoding
    message=$(jq -n --arg file_name "$file_path" --arg file_content "$file_content" '$file_name + "\n" + $file_content')
#    echo "$message"

    json_payload=$(cat << EOF
{
  "message": $message,
  "speaker": "$speaker",
  "timestamp": $file_created_at
}
EOF
)
    echo "$file_path"
    curl -X POST "$web_service_url" \
         -H "Content-Type: application/json" \
         -d "$json_payload"
    echo "Imported"
done < <(find . -type f -name "*.md" -print0)
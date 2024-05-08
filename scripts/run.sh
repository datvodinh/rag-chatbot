
#!/bin/bash

# Define the usage function
usage() {
    echo "Usage: $0 [--ngrok]"
    exit 1
}

# Initialize NGROK variable
NGROK=""

# Loop through command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ngrok)
            NGROK=true
            shift
            ;;
        *)
            usage
            ;;
    esac
done

# Run the Python app
if [[ -n $NGROK ]]; then
    python -m rag_chatbot --host localhost & ngrok http 7860
else
    python -m rag_chatbot --host localhost
fi
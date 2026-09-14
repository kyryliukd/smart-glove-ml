if [[ "$PWD" != *start_server ]]; then
  cd test_server
fi

uvicorn main:app --reload --host 0.0.0.0 --port 8080
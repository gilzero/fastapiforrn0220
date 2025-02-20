## Known Issues

in some cases, the begingin of the response displayed is truncated.




## Test curls

Test the Root Health Check Endpoint (/health)

```bash
curl -X GET "http://localhost:3050/health"
  
```


Test Provider-Specific Health Check (/health/{provider})

```bash
curl -X GET "http://localhost:3050/health/gpt"

curl -X GET "http://localhost:3050/health/claude"

curl -X GET "http://localhost:3050/health/gemini"
  
```



  
  




Test the Chat Endpoint (/chat/{provider}): 

```bash

curl -X POST "http://localhost:3050/chat/gpt" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {"role": "user", "content": "Hello, how are you?"}
           ],
           "model": "gpt-4o"
         }'
         
         
curl -X POST "http://localhost:3050/chat/claude" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {"role": "user", "content": "Hello, how are you?"}
           ],
           "model": "claude-3-5-sonnet-latest"
         }'
         
         
curl -X POST "http://localhost:3050/chat/gemini" \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [
             {"role": "user", "content": "Hello, how are you?"}
           ],
           "model": "gemini-2.0-flash"
         }'

```


## one liner

```bash



```



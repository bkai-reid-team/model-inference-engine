wrk.method = "POST"
wrk.path = "/efficientnet_b0"
wrk.headers["Content-Type"] = "multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"

-- Ảnh JPEG mẫu 1x1 pixel (Base64)
local SAMPLE_IMAGE_B64 = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAD/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFAEBAAAAAAAAAAAAAAAAAAAAAP/EABQRAQAAAAAAAAAAAAAAAAAAAAD/2gAMAwEAAhEDEQAAL+AAf/Z"

-- Giải mã Base64 sang binary (Lua không có sẵn base64 decode, nên dùng ảnh ASCII giả)
local image_data = string.rep("A", 1024)  -- giả ảnh 1KB (đủ để server nhận)

wrk.body = table.concat({
    "------WebKitFormBoundary7MA4YWxkTrZu0gW",
    'Content-Disposition: form-data; name="file"; filename="fake.jpg"',
    "Content-Type: image/jpeg",
    "",
    image_data,
    "------WebKitFormBoundary7MA4YWxkTrZu0gW--",
    ""
}, "\r\n")

section .data
    message: db "Hello, World!", 0Ah  ; Message with newline

section .text
    global _start                      ; Entry point for macOS

_start:
    ; Write to stdout
    mov rax, 0x02000004                ; System call number for write
    mov rdi, 1                         ; File descriptor 1 is stdout
    lea rsi, [rel message]             ; Load effective address of message
    mov rdx, 14                        ; Number of bytes (13 characters + 1 newline)
    syscall                            ; Call kernel

    ; Exit
    mov rax, 0x02000001                ; System call number for exit
    xor rdi, rdi                       ; Exit code 0 (using xor for zero)
    syscall                            ; Call kernel

#include <stdio.h>
#include <unistd.h>

int main(void) {
    printf("hello from pid %d\n", getpid());
    return 0;
}

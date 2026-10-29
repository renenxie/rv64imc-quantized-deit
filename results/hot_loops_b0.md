# 最熱的程式片段（Spike PC 直方圖）

整個程式執行 12,635,450,000 條指令；推論（Invoke）12,617,761,175 條。比例以推論為分母。

## 1. `tflite::micro::{anon}::FullyConnectedEval`
- 每段 8 條指令 × 執行 1,045,949,952 次 = 8,367,599,616 條（66.32%）

```
80001ff0:  lb s6,0(a5)
80001ff4:  lb a4,0(a1)
80001ff8:  addi a5,a5,1
80001ffa:  addw s6,s6,a7
80001ffe:  mulw a4,a4,s6
80002002:  addi a1,a1,1
80002004:  addw a0,a0,a4
80002006:  bne a5,t1,80001ff0
```

## 2. `tflite::micro::{anon}::BatchMatMulEval`
- 每段 11 條指令 × 執行 178,831,872 次 = 1,967,150,592 條（15.59%）

```
80008f24:  lb a0,0(a1)
80008f28:  ld a3,72(sp)
80008f2a:  lb a2,0(a4)
80008f2e:  addi a1,a1,1
80008f30:  subw a0,a0,a3
80008f32:  ld a3,64(sp)
80008f34:  addi a4,a4,1
80008f36:  subw a2,a2,a3
80008f38:  mulw a2,a2,a0
80008f3c:  addw a5,a5,a2
80008f3e:  bne a1,s2,80008f24
```

## 3. `tflite::micro::reference_integer_ops::ConvPerChannel`
- 每段 26 條指令 × 執行 9,633,792 次 = 250,478,592 條（1.99%）

```
8001573e:  not a4,t1
80015742:  srliw a4,a4,0x1f
80015746:  beqz a4,80015734
80015748:  bge t1,s3,80015734
8001574c:  beqz t4,80015734
80015750:  blez s6,80015734
80015754:  lw a6,8(s5)
80015758:  lw a4,8(s4)
8001575c:  ld s7,0(sp)
8001575e:  mulw a6,a6,s8
80015762:  lw a7,12(s5)
80015766:  lw s11,12(s4)
8001576a:  mulw a4,a4,s7
8001576e:  addw a6,a6,t5
80015772:  addw a4,a4,t6
80015776:  mulw a7,a7,a6
8001577a:  lw a6,16(s5)
8001577e:  mulw s11,s11,a4
80015782:  lw a4,16(s4)
80015786:  addw a7,a7,t1
8001578a:  addw s11,s11,t3
8001578e:  mulw s11,a4,s11
80015792:  mulw a7,a6,a7
80015796:  add a4,s10,s11
8001579a:  add s11,s11,s9
8001579c:  add a7,a7,t0
```

## 4. `tflite::micro::FixedPoint<int, 0> tflite::micro::exp_on_negative_values<int, 5>`
- 每段 89 條指令 × 執行 2,770,836 次 = 246,604,404 條（1.95%）

```
80004b66:  addi sp,sp,-32
80004b68:  sd s2,0(sp)
80004b6a:  mv s2,a0
80004b6c:  slli a3,s2,0x28
80004b70:  srli a3,a3,0x28
80004b72:  lui a5,0xff000
80004b76:  sd s1,8(sp)
80004b78:  addw s1,a3,a5
80004b7c:  slliw a0,s1,0x5
80004b80:  sd s0,16(sp)
80004b82:  sd ra,24(sp)
80004b84:  jal 800047be
80004b88:  lui a2,0x63afc
80004b8c:  sext.w a0,a0
80004b8e:  addi a2,a2,-389 # 63afbe7b
80004b92:  mul a2,a0,a2
80004b96:  subw a3,s1,s2
80004b9a:  srli t1,a3,0x18
80004b9e:  andi t1,t1,1
80004ba2:  addiw a5,t1,-1
80004ba6:  and a7,a0,a5
80004baa:  srli a4,a3,0x1e
80004bae:  srli a5,a3,0x19
80004bb2:  srli a6,a3,0x1a
80004bb6:  srli a0,a3,0x1b
80004bba:  srli a1,a3,0x1c
80004bbe:  srli a3,a3,0x1d
80004bc0:  sext.w s0,s2
80004bc4:  andi a4,a4,1
80004bc6:  andi a5,a5,1
80004bc8:  andi a6,a6,1
80004bcc:  andi a0,a0,1
80004bce:  andi a1,a1,1
80004bd0:  andi a3,a3,1
80004bd2:  li t3,0
80004bd4:  bltz a2,80004cf4
80004bd8:  bnez t1,80004ce8
80004bdc:  xor a7,a7,t3
80004be0:  lui a2,0x4da2d
80004be4:  sext.w a7,a7
80004be6:  addi a2,a2,-1038 # 4da2cbf2
80004bea:  mul a2,a7,a2
80004bee:  li t1,0
80004bf0:  bnez a5,80004cdc
80004bf2:  addiw a5,a5,-1 # fffffffffeffffff
80004bf4:  and a5,a7,a5
80004bf8:  xor a5,a5,t1
80004bfc:  lui a2,0x2f16b
80004c00:  sext.w a5,a5
80004c02:  addi a2,a2,-916 # 2f16ac6c
80004c06:  mul a2,a5,a2
80004c0a:  li a7,0
80004c0c:  bnez a6,80004cd0
80004c10:  addiw a6,a6,-1
80004c12:  and a5,a5,a6
80004c16:  xor a5,a5,a7
80004c1a:  lui a2,0x1152b
80004c1e:  sext.w a5,a5
80004c20:  addi a2,a2,-1372 # 1152aaa4
80004c24:  mul a2,a5,a2
80004c28:  li a6,0
80004c2a:  bnez a0,80004cc4
80004c2c:  addiw a0,a0,-1
80004c2e:  and a5,a5,a0
80004c30:  xor a5,a5,a6
80004c34:  lui a2,0x2583
80004c38:  sext.w a5,a5
80004c3a:  addi a2,a2,-1353 # 2582ab7
80004c3e:  mul a2,a5,a2
80004c42:  li a0,0
80004c44:  bnez a1,80004cb8
80004c46:  addiw a1,a1,-1
80004c48:  and a5,a5,a1
80004c4a:  xor a5,a5,a0
80004c4c:  lui a2,0xb0
80004c50:  sext.w a5,a5
80004c52:  addi a2,a2,-495 # afe11
80004c56:  mul a2,a5,a2
80004c5a:  li a1,0
80004c5c:  bnez a3,80004cac
80004c5e:  addiw a3,a3,-1
80004c60:  and a5,a5,a3
80004c62:  xor a5,a5,a1
80004c64:  sext.w a5,a5
80004c66:  li a2,242
80004c6a:  lui a1,0x40000
80004c6e:  mul a2,a5,a2
80004c72:  li a3,0
80004c74:  beqz a4,80004c84
```

## 5. `tflite::micro::reference_integer_ops::ConvPerChannel`
- 每段 8 條指令 × 執行 28,901,376 次 = 231,211,008 條（1.83%）

```
8001579e:  lb s7,0(a7)
800157a2:  lb a6,0(a4)
800157a6:  addi a4,a4,1
800157a8:  addw s7,s7,s0
800157ac:  mulw a6,a6,s7
800157b0:  addi a7,a7,1
800157b2:  addw a0,a6,a0
800157b6:  bne s11,a4,8001579e
```

## 6. `tflite::micro::MultiplyByQuantizedMultiplier`
- 每段 19 條指令 × 執行 6,180,092 次 = 117,421,748 條（0.93%）

```
800154e4:  li a5,1
800154e6:  sll a5,a5,a3
800154ea:  addiw a5,a5,-1 # 7fffffff
800154ec:  sraiw a6,a5,0x1
800154f0:  bnez a4,80015518
800154f2:  add a2,a2,a1
800154f4:  srai a0,a2,0x3f
800154f8:  slli a0,a0,0x21
800154fa:  srli a0,a0,0x21
800154fc:  add a0,a0,a2
800154fe:  srai a0,a0,0x1f
80015500:  srliw a4,a0,0x1f
80015504:  and a5,a5,a0
80015506:  addw a6,a4,a6
8001550a:  sext.w a5,a5
8001550c:  sraw a0,a0,a3
80015510:  slt a5,a6,a5
80015514:  addw a0,a0,a5
80015516:  ret
```

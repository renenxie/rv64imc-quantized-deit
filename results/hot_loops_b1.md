# 最熱的程式片段（Spike PC 直方圖）

整個程式執行 4,404,630,000 條指令；推論（Invoke）4,360,277,226 條。比例以推論為分母。

## 1. `{anon}::FcB1Eval`
- 每段 95 條指令 × 執行 32,514,048 次 = 3,088,834,560 條（70.84%）

```
8001e40e:  lb s10,0(t0) # ffffffff80000000
8001e412:  lb s2,0(a0)
8001e416:  lb t6,0(t2)
8001e41a:  lb s8,0(t3) # 40000000
8001e41e:  mulw s3,s10,s2
8001e422:  lb s7,0(t4)
8001e426:  lb s4,0(t5)
8001e42a:  addi t0,t0,4
8001e42c:  addi t2,t2,4
8001e42e:  addi t5,t5,4
8001e430:  addi a0,a0,4
8001e432:  addi t3,t3,4
8001e434:  addi t4,t4,4
8001e436:  mulw s6,s10,s4
8001e43a:  addw s3,s3,a7
8001e43e:  mulw s4,t6,s4
8001e442:  addw s6,s6,t1
8001e446:  lb t1,-3(t3)
8001e44a:  mulw a7,t6,s8
8001e44e:  addw s4,s4,a1
8001e452:  lb a1,-3(t2)
8001e456:  mulw s2,t6,s2
8001e45a:  addw a7,a7,a4
8001e45e:  lb a4,-3(t0)
8001e462:  mulw t6,t6,s7
8001e466:  addw s2,s2,a3
8001e46a:  lb a3,-3(t5)
8001e46e:  addw a5,t6,a5
8001e472:  lb t6,-3(a0)
8001e476:  mulw s5,s10,s8
8001e47a:  mulw s8,a4,t6
8001e47e:  addw s5,s5,a6
8001e482:  mulw s10,s10,s7
8001e486:  lb s7,-3(t4)
8001e48a:  addw s8,s8,s3
8001e48e:  mulw a6,a4,a3
8001e492:  addw a2,s10,a2
8001e496:  mulw s3,a4,t1
8001e49a:  addw a6,a6,s6
8001e49e:  lb s6,-2(t4)
8001e4a2:  mulw a3,a1,a3
8001e4a6:  addw s3,s3,s5
8001e4aa:  lb s5,-2(a0)
8001e4ae:  mulw t6,a1,t6
8001e4b2:  addw a3,a3,s4
8001e4b6:  lb s4,-2(t3)
8001e4ba:  mulw t1,a1,t1
8001e4be:  addw t6,t6,s2
8001e4c2:  lb s2,-2(t2)
8001e4c6:  mulw a1,a1,s7
8001e4ca:  addw t1,t1,a7
8001e4ce:  addw a1,a1,a5
8001e4d0:  mulw a4,a4,s7
8001e4d4:  lb a5,-2(t0)
8001e4d8:  lb s7,-2(t5)
8001e4dc:  mulw a7,a5,s7
8001e4e0:  addw a4,a4,a2
8001e4e2:  addw a7,a7,a6
8001e4e6:  mulw a2,a5,s4
8001e4ea:  mulw a6,a5,s5
8001e4ee:  addw a2,a2,s3
8001e4f2:  mulw s7,s2,s7
8001e4f6:  addw a6,a6,s8
8001e4fa:  mulw s5,s2,s5
8001e4fe:  addw s7,s7,a3
8001e502:  lb a3,-1(a0)
8001e506:  mulw s4,s2,s4
8001e50a:  addw s5,s5,t6
8001e50e:  lb t6,-1(t0)
8001e512:  mulw s2,s2,s6
8001e516:  addw s4,s4,t1
8001e51a:  addw s2,s2,a1
8001e51e:  lb a1,-1(t5)
8001e522:  mulw a5,a5,s6
8001e526:  lb s6,-1(t4)
8001e52a:  mulw t1,t6,a1
8001e52e:  addw s3,a5,a4
8001e532:  lb a5,-1(t2)
8001e536:  lb a4,-1(t3)
8001e53a:  addw t1,t1,a7
8001e53e:  mulw a7,t6,a3
8001e542:  mulw a1,a5,a1
8001e546:  addw a7,a7,a6
8001e54a:  mulw a3,a5,a3
8001e54e:  addw a1,a1,s7
8001e552:  mulw a6,t6,a4
8001e556:  addw a3,a3,s5
8001e55a:  mulw a4,a5,a4
8001e55e:  addw a6,a6,a2
8001e562:  mulw t6,t6,s6
8001e566:  addw a4,a4,s4
8001e56a:  mulw a5,a5,s6
8001e56e:  addw a2,t6,s3
8001e572:  addw a5,a5,s2
8001e576:  bne t0,s1,8001e40e
```

## 2. `{anon}::BmmB1Eval`
- 每段 95 條指令 × 執行 5,531,904 次 = 525,530,880 條（12.05%）

```
8001ae76:  lb s10,0(t2)
8001ae7a:  lb t0,0(t5)
8001ae7e:  lb t6,0(s0)
8001ae82:  lb s9,0(t1)
8001ae86:  mulw s1,s10,t0
8001ae8a:  lb s8,0(t3)
8001ae8e:  lb s3,0(t4)
8001ae92:  addi t2,t2,4
8001ae94:  addi s0,s0,4
8001ae96:  addi t4,t4,4
8001ae98:  addi t5,t5,4
8001ae9a:  addi t1,t1,4
8001ae9c:  addi t3,t3,4
8001ae9e:  mulw s6,s10,s3
8001aea2:  addw s1,s1,a6
8001aea6:  mulw s3,t6,s3
8001aeaa:  addw s6,s6,a7
8001aeae:  lb a7,-3(t1)
8001aeb2:  mulw a6,t6,s9
8001aeb6:  addw s3,s3,a1
8001aeba:  lb a1,-3(s0)
8001aebe:  mulw t0,t6,t0
8001aec2:  addw a6,a6,a4
8001aec6:  lb a4,-3(t2)
8001aeca:  mulw t6,t6,s8
8001aece:  addw t0,t0,a3
8001aed2:  lb a3,-3(t4)
8001aed6:  addw a5,t6,a5
8001aeda:  lb t6,-3(t5)
8001aede:  mulw s4,s10,s9
8001aee2:  mulw s9,a4,t6
8001aee6:  addw s4,s4,a0
8001aeea:  mulw s10,s10,s8
8001aeee:  lb s8,-3(t3)
8001aef2:  addw s9,s9,s1
8001aef6:  mulw a0,a4,a3
8001aefa:  addw a2,s10,a2
8001aefe:  mulw s1,a4,a7
8001af02:  addw a0,a0,s6
8001af06:  lb s6,-2(t3)
8001af0a:  mulw a3,a1,a3
8001af0e:  addw s1,s1,s4
8001af12:  lb s4,-2(t5)
8001af16:  mulw t6,a1,t6
8001af1a:  addw a3,a3,s3
8001af1e:  lb s3,-2(t1)
8001af22:  mulw a7,a1,a7
8001af26:  addw t6,t6,t0
8001af2a:  lb t0,-2(s0)
8001af2e:  mulw a1,a1,s8
8001af32:  addw a7,a7,a6
8001af36:  addw a1,a1,a5
8001af38:  mulw a4,a4,s8
8001af3c:  lb a5,-2(t2)
8001af40:  lb s8,-2(t4)
8001af44:  mulw a6,a5,s8
8001af48:  addw a4,a4,a2
8001af4a:  addw a6,a6,a0
8001af4e:  mulw a2,a5,s3
8001af52:  mulw a0,a5,s4
8001af56:  addw a2,a2,s1
8001af58:  mulw s8,t0,s8
8001af5c:  addw a0,a0,s9
8001af60:  mulw s4,t0,s4
8001af64:  addw s8,s8,a3
8001af68:  lb a3,-1(t5)
8001af6c:  mulw s3,t0,s3
8001af70:  addw s4,s4,t6
8001af74:  lb t6,-1(t2)
8001af78:  mulw t0,t0,s6
8001af7c:  addw s3,s3,a7
8001af80:  addw t0,t0,a1
8001af84:  lb a1,-1(t4)
8001af88:  mulw a5,a5,s6
8001af8c:  lb s6,-1(t3)
8001af90:  mulw a7,t6,a1
8001af94:  addw s1,a5,a4
8001af98:  lb a5,-1(s0)
8001af9c:  lb a4,-1(t1)
8001afa0:  addw a7,a7,a6
8001afa4:  mulw a6,t6,a3
8001afa8:  mulw a1,a5,a1
8001afac:  addw a6,a6,a0
8001afb0:  mulw a3,a5,a3
8001afb4:  addw a1,a1,s8
8001afb8:  mulw a0,t6,a4
8001afbc:  addw a3,a3,s4
8001afc0:  mulw a4,a5,a4
8001afc4:  addw a0,a0,a2
8001afc6:  mulw t6,t6,s6
8001afca:  addw a4,a4,s3
8001afce:  mulw a5,a5,s6
8001afd2:  addw a2,t6,s1
8001afd6:  addw a5,a5,t0
8001afda:  bne t2,s2,8001ae76
```

## 3. `{anon}::ConvB1Eval`
- 每段 96 條指令 × 執行 903,168 次 = 86,704,128 條（1.99%）

```
8001c2c0:  lb s11,0(t0)
8001c2c4:  lb s3,0(t3)
8001c2c8:  lb t5,0(t2)
8001c2cc:  lb s9,0(t6)
8001c2d0:  mulw s4,s11,s3
8001c2d4:  lb s8,0(t4)
8001c2d8:  lb s5,0(t1) # 40000000
8001c2dc:  addi t6,t6,4
8001c2de:  addi t0,t0,4
8001c2e0:  addi t2,t2,4
8001c2e2:  addi t1,t1,4
8001c2e4:  addi t3,t3,4
8001c2e6:  addi t4,t4,4
8001c2e8:  mulw s7,s11,s5
8001c2ec:  addw s4,s4,a6
8001c2f0:  mulw s5,t5,s5
8001c2f4:  addw s7,s7,a7
8001c2f8:  lb a7,-3(t6)
8001c2fc:  mulw a6,t5,s9
8001c300:  addw s5,s5,a1
8001c304:  lb a1,-3(t2)
8001c308:  mulw s3,t5,s3
8001c30c:  addw a6,a6,a4
8001c310:  lb a4,-3(t0)
8001c314:  mulw t5,t5,s8
8001c318:  addw s3,s3,a3
8001c31c:  lb a3,-3(t1)
8001c320:  addw a5,t5,a5
8001c324:  lb t5,-3(t3)
8001c328:  mulw s6,s11,s9
8001c32c:  mulw s9,a4,t5
8001c330:  addw s6,s6,a0
8001c334:  mulw s11,s11,s8
8001c338:  lb s8,-3(t4)
8001c33c:  addw s9,s9,s4
8001c340:  mulw a0,a4,a3
8001c344:  addw a2,s11,a2
8001c348:  mulw s4,a4,a7
8001c34c:  addw a0,a0,s7
8001c350:  lb s7,-2(t4)
8001c354:  mulw a3,a1,a3
8001c358:  addw s4,s4,s6
8001c35c:  lb s6,-2(t3)
8001c360:  mulw t5,a1,t5
8001c364:  addw a3,a3,s5
8001c368:  lb s5,-2(t6)
8001c36c:  mulw a7,a1,a7
8001c370:  addw t5,t5,s3
8001c374:  lb s3,-2(t2)
8001c378:  mulw a1,a1,s8
8001c37c:  addw a7,a7,a6
8001c380:  addw a1,a1,a5
8001c382:  mulw a4,a4,s8
8001c386:  lb a5,-2(t0)
8001c38a:  lb s8,-2(t1)
8001c38e:  mulw a6,a5,s8
8001c392:  addw a4,a4,a2
8001c394:  addw a6,a6,a0
8001c398:  mulw a2,a5,s5
8001c39c:  mulw a0,a5,s6
8001c3a0:  addw a2,a2,s4
8001c3a4:  mulw s8,s3,s8
8001c3a8:  addw a0,a0,s9
8001c3ac:  mulw s6,s3,s6
8001c3b0:  addw s8,s8,a3
8001c3b4:  lb a3,-1(t3)
8001c3b8:  mulw s5,s3,s5
8001c3bc:  addw s6,s6,t5
8001c3c0:  lb t5,-1(t0)
8001c3c4:  mulw s3,s3,s7
8001c3c8:  addw s5,s5,a7
8001c3cc:  addw s3,s3,a1
8001c3d0:  lb a1,-1(t1)
8001c3d4:  mulw a5,a5,s7
8001c3d8:  lb s7,-1(t4)
8001c3dc:  mulw a7,t5,a1
8001c3e0:  addw s4,a5,a4
8001c3e4:  lb a5,-1(t2)
8001c3e8:  lb a4,-1(t6)
8001c3ec:  addw a7,a7,a6
8001c3f0:  mulw a6,t5,a3
8001c3f4:  mulw a1,a5,a1
8001c3f8:  addw a6,a6,a0
8001c3fc:  mulw a3,a5,a3
8001c400:  addw a1,a1,s8
8001c404:  mulw a0,t5,a4
8001c408:  addw a3,a3,s6
8001c40c:  mulw a4,a5,a4
8001c410:  addw a0,a0,a2
8001c412:  mulw t5,t5,s7
8001c416:  addw a4,a4,s5
8001c41a:  mv s5,s1
8001c41c:  mulw a5,a5,s7
8001c420:  addw a2,t5,s4
8001c424:  addw a5,a5,s3
8001c428:  bne t6,s1,8001c2c0
```

## 4. `tflite::micro::{anon}::MulB1Eval`
- 每段 20 條指令 × 執行 4,233,924 次 = 84,678,480 條（1.94%）

```
8001d61c:  sb a4,0(t1)
8001d620:  addi t1,t1,1
8001d622:  add s0,s0,s4
8001d624:  add t0,t0,s6
8001d626:  beq a5,t1,8001d68c
8001d62a:  lb a2,0(t0)
8001d62e:  lb a4,0(s0)
8001d632:  lui a0,0x40000
8001d636:  subw a2,a2,s3
8001d63a:  subw a4,a4,s2
8001d63e:  mulw a4,a4,a2
8001d642:  lui a2,0x80000
8001d646:  sllw a4,a4,a3
8001d64a:  sub t5,a4,a2
8001d64e:  sub a2,a7,a4
8001d652:  mul a4,a4,a7
8001d656:  seqz t5,t5
8001d65a:  seqz a2,a2
8001d65e:  and a2,t5,a2
8001d662:  bgez a4,8001d66c
```

## 5. `tflite::micro::TfLiteStatus tflite::micro::{anon}::AddSubEval<tflite::micro::OpDataAdd, false>`
- 每段 22 條指令 × 執行 1,896,125 次 = 41,714,750 條（0.96%）

```
8001d830:  sb a5,0(a0)
8001d834:  addi a0,a0,1
8001d836:  add a3,a3,s1
8001d838:  add a2,a2,t2
8001d83a:  beq a1,a0,8001d8a0
8001d83e:  lb a4,0(a2) # ffffffff80000000
8001d842:  lb a5,0(a3)
8001d846:  slli a4,a4,0x2
8001d848:  slli a5,a5,0x2
8001d84a:  add a4,a4,s0
8001d84c:  add a5,a5,s0
8001d84e:  lw a4,1596(a4)
8001d852:  lw a5,572(a5)
8001d856:  addw a5,a5,a4
8001d858:  sub a4,a5,t0
8001d85c:  sub s6,a6,a5
8001d860:  mul a5,a5,a6
8001d864:  seqz s6,s6
8001d868:  seqz a4,a4
8001d86c:  and a4,a4,s6
8001d870:  lui s6,0x40000
8001d874:  bgez a5,8001d87e
```

## 6. `tflite::micro::{anon}::MulB1Eval`
- 每段 8 條指令 × 執行 4,233,924 次 = 33,871,392 條（0.78%）

```
8001d5f6:  sext.w t5,t5
8001d5f8:  addw a2,a0,t2
8001d5fc:  slt a2,a2,t5
8001d600:  sraw a4,a4,a1
8001d604:  addw a4,a4,a2
8001d606:  addw a4,a4,s5
8001d60a:  mv a2,a4
8001d60c:  bge a4,t4,8001d612
```

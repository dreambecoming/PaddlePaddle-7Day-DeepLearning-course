```mermaid
graph LR
  subgraph g1
    a1-->b1
  end
  subgraph g2
    a2-->b2
  end
  subgraph g3
    a3-->b3
  end
  a3-->a2
```

```mermaid
flowchat
st=>start: Start
e=>end: End
op1=>operation: Operation
sub1=>subroutine: Subroutine
cond=>condition: yes or no ?
io=>inputoutput: proceess something...
st->op1->cond`
cond(yes)->io->e`
cond(no)->sub1(right)->op1
```

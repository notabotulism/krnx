# Chillbot Architecture Diagrams

## System Architecture

```mermaid
flowchart TB
    subgraph SDK["SDK Layer"]
        Memory["Memory Class<br/>(Simple Interface)"]
        KClient["KRNXClient<br/>(Infrastructure)"]
        Agent["Agent<br/>(Multi-agent)"]
    end

    subgraph Fabric["Memory Fabric"]
        Orch["Orchestrator"]
        Context["Context Builder"]
        Identity["Identity Resolver"]
        Retention["Retention Policies"]
    end

    subgraph Compute["KRNX Compute"]
        Queue["Job Queue<br/>(SQLite)"]
        Worker["Worker Loop"]
        Embed["Embeddings<br/>(sentence-transformers)"]
        Vectors["Vector Store<br/>(Qdrant)"]
        Salience["Salience Scoring"]
    end

    subgraph Kernel["KRNX Kernel"]
        EventStore["Event Store<br/>(append-only)"]
        STM["STM<br/>(Redis)"]
        LTM["LTM<br/>(SQLite WAL)"]
        Replay["Replay Engine"]
        Hash["Hash Chain"]
    end

    subgraph Storage["Storage Layer"]
        Redis[(Redis/KeyDB)]
        SQLite[(SQLite WAL)]
        Qdrant[(Qdrant)]
        S3[(S3/MinIO)]
    end

    Memory --> Orch
    KClient --> Kernel
    Agent --> Kernel

    Orch --> Kernel
    Orch --> Queue
    Orch --> Vectors
    Context --> Vectors
    Context --> Kernel

    Queue --> Worker
    Worker --> Embed
    Worker --> Vectors
    Worker --> Salience

    STM --> Redis
    LTM --> SQLite
    Vectors --> Qdrant
    EventStore --> S3
```

## Data Flow: Remember

```mermaid
sequenceDiagram
    participant App
    participant Memory
    participant Fabric
    participant Kernel
    participant Queue
    participant Worker
    participant Qdrant

    App->>Memory: remember("user likes hiking")
    Memory->>Fabric: remember(workspace, user, content)
    Fabric->>Kernel: write_event(event)
    Kernel-->>Fabric: event_id
    Fabric->>Queue: enqueue(EMBED, event_id, text)
    Queue-->>Fabric: job_id
    Fabric-->>Memory: event_id
    Memory-->>App: event_id

    Note over Worker: Background processing
    Worker->>Queue: dequeue()
    Queue-->>Worker: job
    Worker->>Worker: embed(text)
    Worker->>Qdrant: index(event_id, vector)
    Worker->>Queue: complete(job_id)
```

## Data Flow: Recall

```mermaid
sequenceDiagram
    participant App
    participant Memory
    participant Fabric
    participant Embed
    participant Qdrant
    participant Kernel

    App->>Memory: recall("outdoor activities")
    Memory->>Fabric: recall(workspace, query)
    Fabric->>Embed: embed(query)
    Embed-->>Fabric: query_vector
    Fabric->>Qdrant: search(query_vector, top_k)
    Qdrant-->>Fabric: matches (event_ids + scores)
    Fabric->>Kernel: get_events(event_ids)
    Kernel-->>Fabric: events
    Fabric->>Fabric: merge & rank
    Fabric-->>Memory: enriched memories
    Memory-->>App: memories
```

## Component Ownership

```mermaid
flowchart LR
    subgraph Kernel["Kernel Owns"]
        E1[Events]
        E2[Hash Chain]
        E3[STM State]
        E4[LTM State]
        E5[Replay Logic]
    end

    subgraph Compute["Compute Owns"]
        C1[Job Queue]
        C2[Worker Pool]
        C3[Vectors]
        C4[Embeddings]
        C5[Salience Scores]
    end

    subgraph Fabric["Fabric Owns"]
        F1[Identity Registry]
        F2[Context Assembly]
        F3[Retention Config]
        F4[Read Orchestration]
    end

    Kernel -->|events| Compute
    Compute -->|vectors/scores| Fabric
    Kernel -->|raw events| Fabric
```

## File Structure

```
chillbot/
├── __init__.py                 # Memory, Agent exports
├── memory.py                   # Simple interface ← YOU ARE HERE NEXT
│
├── kernel/                     # YOUR EXISTING KRNX CODE
│   ├── __init__.py             
│   ├── client.py               # ← krnx_sdk/client.py
│   ├── agent.py                # ← krnx_sdk/agent.py
│   ├── exceptions.py           # ← krnx_sdk/exceptions.py
│   ├── controller.py           # ← krnx/controller.py
│   ├── stm.py                  # ← krnx/stm.py
│   ├── ltm.py                  # ← krnx/ltm.py
│   └── models.py               # ← krnx/models.py
│
├── compute/                    # ✅ COMPLETE
│   ├── __init__.py             
│   ├── queue.py                # SQLite job queue
│   ├── embeddings.py           # sentence-transformers
│   ├── vectors.py              # Qdrant interface
│   ├── salience.py             # Importance scoring
│   └── worker.py               # Background processor
│
├── fabric/                     # 🔄 NEXT TO BUILD
│   ├── __init__.py             
│   ├── orchestrator.py         # Main MemoryFabric class
│   ├── context.py              # Context builder
│   ├── identity.py             # Agent/scope resolution
│   └── retention.py            # Retention policy triggers
│
└── server/                     # API SERVER
    ├── __init__.py             
    ├── api.py                  # ← krnx/api_server.py
    └── config.py               
```

## Deployment Topology

```mermaid
flowchart TB
    subgraph Client["Client Applications"]
        SDK1["Python SDK"]
        SDK2["Node SDK"]
        HTTP["HTTP Clients"]
    end

    subgraph Gateway["API Gateway"]
        API["FastAPI Server<br/>:6380"]
    end

    subgraph Workers["Compute Workers"]
        W1["Worker 1"]
        W2["Worker 2"]
        W3["Worker N"]
    end

    subgraph Data["Data Stores"]
        Redis[("Redis<br/>:6379<br/>STM")]
        SQLite[("SQLite<br/>LTM + Jobs")]
        Qdrant[("Qdrant<br/>:6333<br/>Vectors")]
        MinIO[("MinIO<br/>:9000<br/>Backups")]
    end

    SDK1 --> API
    SDK2 --> API
    HTTP --> API

    API --> Redis
    API --> SQLite

    W1 --> SQLite
    W2 --> SQLite
    W3 --> SQLite

    W1 --> Qdrant
    W2 --> Qdrant
    W3 --> Qdrant

    SQLite -.->|backup| MinIO
    Redis -.->|backup| MinIO
    Qdrant -.->|backup| MinIO
```

## Hosting Phases

```mermaid
gantt
    title Chillbot Development Roadmap
    dateFormat  YYYY-MM-DD
    section Substrate
    Kernel (Complete)           :done,    k1, 2024-01-01, 2024-02-15
    Kernel SDK (Complete)       :done,    k2, 2024-02-01, 2024-02-20
    Compute (Complete)          :done,    c1, 2024-03-01, 2024-03-10
    Fabric                      :active,  f1, 2024-03-10, 2024-03-20
    Memory Class                :         m1, 2024-03-15, 2024-03-25
    
    section Hosting v1
    Control Plane               :         h1, 2024-03-20, 2024-04-05
    Primitive Hosting           :         h2, 2024-03-25, 2024-04-10
    Full Stack Hosting          :         h3, 2024-04-01, 2024-04-15
    
    section Launch
    Documentation               :         d1, 2024-04-01, 2024-04-10
    OSS Release                 :         o1, 2024-04-10, 2024-04-15
    HN Launch                   :milestone, 2024-04-15, 0d
```

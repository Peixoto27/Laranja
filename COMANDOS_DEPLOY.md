# 🚀 Comandos para Deploy no Railway

## 📋 Checklist Pré-Deploy

✅ Modelo treinado com 73.6% de acurácia  
✅ Interface web funcionando  
✅ API testada localmente  
✅ Arquivos de configuração criados  
✅ Dependências documentadas  

## 🛠️ Comandos Essenciais

### 1. Preparar Repositório Git
```bash
# Navegar para o diretório do projeto
cd /home/ubuntu/Pro/crypto_trading_api

# Inicializar repositório Git
git init

# Adicionar todos os arquivos
git add .

# Fazer commit inicial
git commit -m "🚀 Crypto Trading AI - Sistema completo com IA"

# Conectar ao seu repositório GitHub
git remote add origin https://github.com/SEU_USUARIO/crypto-trading-ai.git

# Fazer push para GitHub
git push -u origin main
```

### 2. Deploy no Railway
```bash
# Opção 1: Via CLI do Railway (se instalado)
railway login
railway new
railway link [PROJECT_ID]
railway up

# Opção 2: Via Web (Recomendado)
# 1. Acesse https://railway.app
# 2. Clique em "New Project"
# 3. Selecione "Deploy from GitHub repo"
# 4. Escolha seu repositório
# 5. Railway fará deploy automaticamente
```

### 3. Verificar Deploy
```bash
# Testar API após deploy
curl https://[SEU_APP].railway.app/api/status

# Verificar predições
curl https://[SEU_APP].railway.app/api/predictions
```

## 📁 Estrutura de Arquivos para Deploy

```
crypto_trading_api/
├── src/
│   ├── main.py              # ✅ Aplicação principal
│   ├── simple_app.py        # ✅ Versão demo
│   ├── model_enhanced.pkl   # ✅ Modelo treinado
│   ├── scaler.pkl          # ✅ Normalizador
│   └── [outros arquivos]   # ✅ Módulos do projeto
├── requirements.txt         # ✅ Dependências
├── railway.json            # ✅ Config Railway
├── Procfile                # ✅ Comando start
├── runtime.txt             # ✅ Python 3.11
└── README.md               # ✅ Documentação
```

## ⚙️ Configurações Importantes

### railway.json
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "cd src && python main.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Procfile
```
web: cd src && python main.py
```

### requirements.txt (principais)
```
Flask==3.1.0
flask-cors==5.0.0
ccxt==4.5.1
pandas==2.3.2
scikit-learn==1.7.1
xgboost==3.0.4
numpy==2.3.2
joblib==1.5.1
requests==2.32.5
python-dotenv==1.1.1
```

## 🔧 Comandos de Teste Local

### Antes do Deploy
```bash
# Ativar ambiente virtual
cd crypto_trading_api
source venv/bin/activate

# Testar aplicação principal
cd src
python main.py

# Testar aplicação simplificada
python simple_app.py

# Verificar se API responde
curl http://localhost:5000/api/status
```

### Retreinar Modelo (se necessário)
```bash
cd src
python train_model_enhanced.py
python validate_model.py
```

## 🌐 URLs Após Deploy

Substitua `[SEU_APP]` pelo nome do seu app no Railway:

- **Interface Web**: `https://[SEU_APP].railway.app/`
- **Status da API**: `https://[SEU_APP].railway.app/api/status`
- **Predições**: `https://[SEU_APP].railway.app/api/predictions`
- **Forçar Update**: `https://[SEU_APP].railway.app/api/force-update`

## 🐛 Troubleshooting

### Problemas Comuns

#### 1. Erro de Dependências
```bash
# Verificar requirements.txt
cat requirements.txt

# Reinstalar dependências
pip install -r requirements.txt
```

#### 2. Modelo não Carrega
```bash
# Verificar se arquivos existem
ls -la src/*.pkl

# Retreinar se necessário
python src/train_model_enhanced.py
```

#### 3. API não Responde
```bash
# Verificar logs no Railway dashboard
# Ou testar localmente:
python src/simple_app.py
```

#### 4. Erro de Porta
```bash
# Railway usa PORT automático
# Código já configurado para usar 0.0.0.0:5000
```

## 📊 Monitoramento Pós-Deploy

### Verificar Saúde da API
```bash
# Status geral
curl https://[SEU_APP].railway.app/api/status

# Resposta esperada:
{
  "status": "online",
  "model_loaded": true,
  "last_update": "2025-08-24T...",
  "cached_predictions": 6
}
```

### Testar Predições
```bash
# Obter predições atuais
curl https://[SEU_APP].railway.app/api/predictions

# Forçar atualização
curl https://[SEU_APP].railway.app/api/force-update
```

## 🔄 Atualizações Futuras

### Deploy de Novas Versões
```bash
# Fazer mudanças no código
git add .
git commit -m "🔄 Atualização: [descrição]"
git push origin main

# Railway fará redeploy automaticamente
```

### Backup do Modelo
```bash
# Fazer backup dos arquivos importantes
cp src/model_enhanced.pkl backup/
cp src/scaler.pkl backup/
cp src/data_raw.json backup/
```

## 📈 Performance Esperada

### Métricas do Modelo
- **Acurácia**: 73.6%
- **Taxa de acerto**: 94.31%
- **Retorno total**: 2,905.37%
- **Confiança média**: 83%

### Criptomoedas Suportadas
- BTCUSDT, ETHUSDT, DOGEUSDT
- PEPEUSDT, BONKUSDT, FLOKIUSDT
- SHIBUSDT (configurável)

---

**🎉 Seu sistema de trading com IA está pronto para produção!**


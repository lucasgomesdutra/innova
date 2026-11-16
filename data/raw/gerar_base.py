"""
=============================================================================
DETECTOR DE PROBLEMAS NO ATENDIMENTO — SICREDI
data/raw/gerar_base.py — Gerador da base fictícia de atendimentos
=============================================================================

Este script gera o arquivo atendimentos.csv com dados sintéticos para
treinamento e teste dos modelos. Os dados são completamente fictícios.
"""

import pandas as pd
import random
from datetime import datetime, timedelta

# Semente para reprodutibilidade
random.seed(42)

# =============================================================================
# BANCO DE TEXTOS POR CATEGORIA
# =============================================================================

TEXTOS_POR_CATEGORIA = {

    "Problemas no aplicativo": [
        "Não consigo entrar no aplicativo.",
        "O app não abre quando tento acessar.",
        "Estou tendo dificuldades para acessar minha conta pelo aplicativo.",
        "O aplicativo trava toda vez que abro.",
        "App fecha sozinho sem motivo aparente.",
        "O aplicativo da Sicredi parou de funcionar no meu celular.",
        "Não consigo abrir o app, fica carregando e não entra.",
        "O aplicativo ficou lento e não carrega as informações.",
        "Tela do aplicativo ficou preta após a última atualização.",
        "Depois de atualizar o app, não consigo mais fazer login.",
        "O aplicativo apresenta erro ao tentar abrir.",
        "Erro 500 aparece toda vez que tento usar o aplicativo.",
        "O app não funciona no meu celular novo.",
        "Aplicativo não abre desde ontem.",
        "Tentei usar o aplicativo várias vezes e não carrega.",
        "O aplicativo está fora do ar.",
        "Não aparece saldo no aplicativo.",
        "O aplicativo não atualiza o extrato.",
        "Função de pagamento do aplicativo não funciona.",
        "O app apresenta mensagem de erro ao tentar acessar.",
        "Meu aplicativo Sicredi não sincroniza.",
        "O aplicativo demora demais para carregar.",
        "Não consigo instalar o aplicativo no meu celular.",
        "Aplicativo fica pedindo atualização mas não atualiza.",
        "O aplicativo não reconhece minha senha.",
        "O login biométrico parou de funcionar no app.",
        "App não abre após atualização do sistema operacional.",
        "O aplicativo está com problema de conexão.",
        "Recebi mensagem de erro inesperado no aplicativo.",
        "O app não mostra meus cartões cadastrados.",
    ],

    "Problemas com Pix e transferências": [
        "Fiz um Pix e o dinheiro não chegou para o destinatário.",
        "Meu Pix foi debitado mas não foi recebido.",
        "Não consigo cadastrar uma chave Pix.",
        "A transferência foi estornada sem motivo.",
        "Pix ficou pendente há mais de dois dias.",
        "Não consigo realizar transferências pelo aplicativo.",
        "Erro ao tentar enviar Pix.",
        "O Pix foi cancelado automaticamente.",
        "Tentei fazer uma transferência TED e não foi.",
        "A transferência sumiu da minha conta.",
        "Cobrado duas vezes pelo mesmo Pix.",
        "Pix enviado para chave errada, quero estorno.",
        "Não consegui cancelar uma transferência agendada.",
        "Minha transferência foi recusada sem explicação.",
        "Não encontro a opção de Pix no aplicativo.",
        "Chave Pix não está sendo aceita.",
        "Pix não chegou na conta do destinatário.",
        "Fiz transferência e o valor foi debitado duas vezes.",
        "Preciso de comprovante de Pix que não aparece.",
        "Não consigo fazer Pix para outro banco.",
        "O sistema recusou minha transferência bancária.",
        "Erro ao tentar fazer TED para conta externa.",
        "Não aparece o histórico de Pix no aplicativo.",
        "Quero cancelar um Pix agendado mas não consigo.",
        "Meu Pix foi devolvido sem justificativa.",
        "Não consigo cadastrar conta para transferência.",
        "O limite de Pix noturno está bloqueando minha transferência.",
        "Tentei pagar uma conta por transferência e deu erro.",
        "A transferência está aparecendo como processando há horas.",
        "Não recebi confirmação do Pix enviado.",
    ],

    "Cobranças indevidas": [
        "Fui cobrado por um serviço que não contratei.",
        "Há uma cobrança estranha na minha fatura.",
        "Cobraram minha conta duas vezes pelo mesmo serviço.",
        "Recebi uma taxa que não estava prevista.",
        "Estão cobrando tarifas que não concordei.",
        "Apareceu uma cobrança que não reconheço no extrato.",
        "Fui debitado por um seguro que não contratei.",
        "A cobrança de anuidade foi feita incorretamente.",
        "Me cobraram juros indevidos.",
        "Estornaram um valor errado da minha conta.",
        "Tarifa de manutenção cobrada indevidamente.",
        "Cobrado por serviço que já havia cancelado.",
        "Não deveria ter sido cobrado essa taxa.",
        "Cobrou por dois meses no mesmo dia.",
        "Cobrança duplicada no mesmo extrato.",
        "O valor debitado está diferente do combinado.",
        "Me cobraram mais do que deveria.",
        "Descontaram um valor da minha conta sem autorização.",
        "Cobrança indevida de IOF.",
        "Me enviaram boleto com valor maior que o contratado.",
        "O seguro foi cobrado mesmo após o cancelamento.",
        "Estão cobrando tarifa de conta que foi encerrada.",
        "Fui cobrado por limite de crédito que não solicitei.",
        "Recebi desconto em dobro no mesmo mês.",
        "A fatura veio com cobrança duplicada.",
        "Me cobraram por serviço de internet banking sem ter contratado.",
        "A corretagem foi cobrada indevidamente.",
        "Verifico no extrato um débito não autorizado.",
        "Houve estorno indevido de um pagamento que eu fiz.",
        "Valores diferentes dos informados na proposta foram cobrados.",
    ],

    "Demora no atendimento": [
        "Esperei mais de uma hora para ser atendido.",
        "O atendimento demorou muito mais que o esperado.",
        "Fiquei na fila por muito tempo sem ser chamado.",
        "Liguei para o atendimento e fiquei esperando muito tempo.",
        "O tempo de espera na agência é absurdo.",
        "O chat demorou para responder.",
        "Não consigo falar com um atendente.",
        "O suporte não retornou minha solicitação.",
        "Aguardo resposta há três dias.",
        "Minha solicitação está sem resposta há muito tempo.",
        "O atendimento telefônico é muito demorado.",
        "Fica tocando e ninguém atende.",
        "Passei pelo URA e não consegui falar com ninguém.",
        "Esperei 40 minutos na fila da agência.",
        "O sistema de chat não funciona bem.",
        "A fila de espera está sempre muito longa.",
        "O gerente não retorna as ligações.",
        "Meu chamado foi aberto há dias e ninguém me contatou.",
        "Demorou mais de uma semana para resolver meu problema.",
        "Não tenho retorno dos atendentes.",
        "O chatbot não resolve e não passa para humano.",
        "O atendimento presencial está sobrecarregado.",
        "Tentei agendar e não encontrei horário disponível.",
        "O tempo de espera no telefone é inaceitável.",
        "Ninguém responde os meus e-mails enviados ao suporte.",
        "A atendente não sabia resolver meu problema.",
        "Precisei ligar três vezes para resolver uma questão simples.",
        "O atendimento online está muito lento.",
        "Fui direcionado para vários setores sem resolver meu caso.",
        "Não há atendimento disponível nos horários que preciso.",
    ],

    "Problemas de acesso à conta": [
        "Não consigo acessar minha conta bancária.",
        "Minha senha foi bloqueada.",
        "Não lembro minha senha e não consigo redefinir.",
        "O sistema não aceita minha senha eletrônica.",
        "Minha conta foi bloqueada sem motivo.",
        "Não consigo entrar na conta pelo internet banking.",
        "O sistema pede um token que não recebi.",
        "Não consigo recuperar meu acesso.",
        "Meu login está bloqueado.",
        "A conta ficou inacessível após troca de celular.",
        "Não consigo fazer o cadastro de nova senha.",
        "O token não está chegando no meu celular.",
        "Minha conta foi suspensa sem comunicação.",
        "Troquei de número e perdi o acesso ao aplicativo.",
        "O sistema não aceita meu CPF para login.",
        "Preciso desbloquear minha conta com urgência.",
        "Não consigo completar o processo de autenticação.",
        "O reconhecimento facial não funciona.",
        "Perdi acesso à conta após atualizar o celular.",
        "A senha de quatro dígitos não está sendo aceita.",
        "O sistema diz que meu usuário não existe.",
        "Não recebo o código SMS para autenticação.",
        "Meu acesso foi revogado sem motivo.",
        "Não consigo configurar o segundo fator de autenticação.",
        "A conta foi bloqueada por tentativas erradas de senha.",
        "O aplicativo não reconhece minha digital.",
        "Não consigo confirmar minha identidade pelo sistema.",
        "Preciso acessar a conta mas o sistema rejeita minhas credenciais.",
        "Esqueci a senha e o processo de recuperação não funciona.",
        "Meu acesso está sendo negado mesmo com credenciais corretas.",
    ],

    "Problemas com cartões": [
        "Meu cartão foi bloqueado sem justificativa.",
        "O cartão não está sendo aceito nas maquininhas.",
        "Fiz uma compra e o cartão foi recusado.",
        "Meu cartão de crédito foi cancelado indevidamente.",
        "Não recebi meu novo cartão.",
        "O cartão chegou com o nome errado.",
        "Meu cartão foi clonado.",
        "Há compras que não reconheço na fatura.",
        "O limite do cartão está errado.",
        "Não consigo desbloquear meu cartão.",
        "O cartão não funciona em compras internacionais.",
        "Meu cartão virtual não aparece no aplicativo.",
        "A fatura do cartão está incorreta.",
        "Não consigo contestar uma compra no cartão.",
        "O cartão foi expirado antes do prazo.",
        "O chip do cartão não funciona.",
        "Compras foram feitas com meu cartão sem minha autorização.",
        "Não estou recebendo notificações de uso do cartão.",
        "O cartão foi debitado mas a compra não foi confirmada.",
        "Não consigo ver meu limite disponível.",
        "O cartão foi bloqueado após viagem.",
        "A senha do cartão não está sendo aceita.",
        "Solicitar segunda via do cartão está muito difícil.",
        "Meu cartão novo não chegou depois de 20 dias.",
        "O cartão não passa no débito.",
        "Transação foi aprovada mas o produto não foi entregue.",
        "Meu cartão foi cancelado sem aviso prévio.",
        "Não consigo usar o cartão para pagamentos recorrentes.",
        "O cartão não funciona pelo Google Pay.",
        "Tive o cartão furtado e o bloqueio foi demorado.",
    ],

    "Dificuldades com serviços bancários": [
        "Não consigo fazer um empréstimo pelo sistema.",
        "Minha solicitação de crédito foi negada sem justificativa.",
        "Tive dificuldades para abrir minha conta.",
        "O processo de contratação de seguro é muito complicado.",
        "Não entendo como funciona meu investimento.",
        "Quero cancelar um serviço mas não consigo.",
        "O processo de portabilidade está travado.",
        "Não consigo assinar contratos digitalmente.",
        "Os juros do financiamento são diferentes do informado.",
        "Preciso de renegociação mas o sistema não permite.",
        "Não consigo emitir boleto pelo aplicativo.",
        "O saldo do consórcio não atualiza corretamente.",
        "A proposta de financiamento sumiu do sistema.",
        "Não consigo visualizar meu histórico de transações.",
        "Tive problemas para contratar o seguro de vida.",
        "O processo de solicitação de limite é burocrático demais.",
        "Quero fazer portabilidade de salário mas não sei como.",
        "O extrato bancário não está correto.",
        "Tenho dificuldade em usar o serviço de débito automático.",
        "Preciso de informações sobre minha previdência privada.",
        "Não consigo agendar pagamentos futuros.",
        "A simulação de crédito apresenta valores inconsistentes.",
        "Não entendo as taxas cobradas no meu financiamento.",
        "Quero cancelar o débito automático mas não encontro a opção.",
        "Preciso de uma declaração bancária mas não consigo emitir.",
        "A conta poupança não está rendendo como esperado.",
        "Não recebi o comprovante de abertura de conta.",
        "O processo de encerramento de conta é complicado.",
        "Não consigo atualizar meus dados cadastrais.",
        "O sistema não aceita meu endereço para cadastro.",
    ],

    "Elogios e avaliações positivas": [
        "O atendimento foi excelente, fiquei muito satisfeito.",
        "A equipe resolveu meu problema rapidamente.",
        "Ótimo atendimento, parabéns à equipe.",
        "Adorei o novo aplicativo, ficou muito intuitivo.",
        "O atendente foi muito prestativo e resolveu tudo.",
        "Estou muito satisfeito com os serviços da Sicredi.",
        "O aplicativo é fácil de usar e funciona muito bem.",
        "Fui bem atendido na agência, serviço de qualidade.",
        "Resolvi minha questão rapidamente com a ajuda da equipe.",
        "Recomendo a Sicredi pelo atendimento diferenciado.",
        "Gostei muito da forma como fui atendido.",
        "O suporte online é muito eficiente.",
        "Excelente experiência com o internet banking.",
        "A Sicredi sempre supera minhas expectativas.",
        "Muito bom o atendimento pelo aplicativo.",
        "Conta com tarifa justa e ótimos serviços.",
        "Me sinto bem atendido como cooperado.",
        "O gerente foi muito atencioso e prestativo.",
        "Processo de financiamento foi simples e rápido.",
        "Estou satisfeito com as taxas e serviços oferecidos.",
        "O novo sistema de Pix ficou muito prático.",
        "Adoro ser cooperado da Sicredi.",
        "O aplicativo melhorou muito com a atualização.",
        "Atendimento humanizado e eficiente.",
        "Fui bem orientado sobre meus investimentos.",
        "A cooperativa se preocupa com o cliente.",
        "Nunca tive problemas com a Sicredi.",
        "O atendimento é sempre cordial e profissional.",
        "Problema resolvido no mesmo dia, parabéns.",
        "A Sicredi tem os melhores atendentes que já vi.",
        # Elogios específicos sobre Pix e transferências
        "Rápido e fácil de fazer transferências pelo aplicativo.",
        "O Pix da Sicredi é muito rápido e prático.",
        "Fazer transferências nunca foi tão simples.",
        "O Pix caiu na hora, perfeito!",
        "Transferência realizada com sucesso e sem complicação.",
        "Adorei a facilidade de fazer Pix pelo aplicativo.",
        "Pix instantâneo e sem custo, excelente serviço.",
        "As transferências funcionam muito bem, sem nenhum problema.",
        "Consigo fazer Pix com facilidade a qualquer hora.",
        "O sistema de transferências é muito ágil e confiável.",
        # Elogios sobre o aplicativo
        "O aplicativo é muito intuitivo e fácil de navegar.",
        "Nunca tive problema com o aplicativo da Sicredi.",
        "O app funciona perfeitamente no meu celular.",
        "Acesso minha conta pelo aplicativo sem nenhuma dificuldade.",
        "O aplicativo carrega rápido e é muito fácil de usar.",
        # Elogios sobre acesso e cartões
        "Meu acesso à conta funciona sempre perfeitamente.",
        "O cartão da Sicredi é aceito em todos os lugares.",
        "Nunca tive problema com meu cartão.",
        "O desbloqueio do cartão foi rápido e simples.",
        "Estou muito satisfeito com os serviços bancários da Sicredi.",
    ],

    "Outros": [
        "Quero saber como abrir uma conta na Sicredi.",
        "Preciso de informações sobre os produtos disponíveis.",
        "Gostaria de entender como funciona a cooperativa.",
        "Tenho interesse em investir mas não sei como começar.",
        "Preciso de um gerente para falar sobre minha conta.",
        "Quero informações sobre consórcio.",
        "Desejo indicar um familiar para abrir conta.",
        "Preciso de uma declaração para imposto de renda.",
        "Gostaria de participar das assembleias da cooperativa.",
        "Como funciona a participação nos resultados?",
        "Quero saber sobre os benefícios do cartão.",
        "Preciso de ajuda para entender meu extrato.",
        "Tenho uma sugestão de melhoria para o aplicativo.",
        "Gostaria de mudar de agência.",
        "Preciso de informações sobre previdência privada.",
        "Como funciona o seguro residencial da Sicredi?",
        "Quero saber mais sobre o programa de pontos.",
        "Preciso de informações sobre crédito rural.",
        "Gostaria de entender os termos do contrato.",
        "Tenho dúvidas sobre a cobrança de IOF.",
        "Como cancelo um serviço contratado?",
        "Quero atualizar meu endereço no cadastro.",
        "Preciso emitir uma certidão negativa de débitos.",
        "Gostaria de informações sobre os seguros oferecidos.",
        "Como faço para aumentar meu limite de crédito?",
        "Quero entender meu extrato do consórcio.",
        "Preciso de segunda via do boleto de mensalidade.",
        "Como funciona a portabilidade de crédito?",
        "Tenho interesse em financiamento imobiliário.",
        "Gostaria de falar sobre o seguro de vida.",
    ],
}

# Mapeamento de sentimento por categoria
SENTIMENTO_POR_CATEGORIA = {
    "Problemas no aplicativo": "negativo",
    "Problemas com Pix e transferências": "negativo",
    "Cobranças indevidas": "negativo",
    "Demora no atendimento": "negativo",
    "Problemas de acesso à conta": "negativo",
    "Problemas com cartões": "negativo",
    "Dificuldades com serviços bancários": "negativo",
    "Elogios e avaliações positivas": "positivo",
    "Outros": "neutro",
}

# =============================================================================
# GERAÇÃO DOS REGISTROS
# =============================================================================

def gerar_data_aleatoria(inicio="2024-01-01", fim="2024-12-31"):
    """Gera uma data aleatória entre dois limites."""
    fmt = "%Y-%m-%d"
    inicio_dt = datetime.strptime(inicio, fmt)
    fim_dt = datetime.strptime(fim, fmt)
    delta = fim_dt - inicio_dt
    return (inicio_dt + timedelta(days=random.randint(0, delta.days))).strftime(fmt)


def gerar_nota(sentimento):
    """Gera uma nota de satisfação coerente com o sentimento."""
    if sentimento == "positivo":
        return random.choice([4, 5, 5, 5])
    elif sentimento == "negativo":
        return random.choice([1, 1, 2, 2, 3])
    else:
        return random.choice([3, 3, 4])


def gerar_canal():
    """Gera o canal de origem do atendimento."""
    return random.choice(["aplicativo", "telefone", "chat", "agência", "email"])


def gerar_base(repetir=3):
    """
    Gera a base completa de atendimentos fictícios.
    Cada texto da base é usado `repetir` vezes para aumentar o volume.
    """
    registros = []
    id_counter = 1

    for categoria, textos in TEXTOS_POR_CATEGORIA.items():
        sentimento = SENTIMENTO_POR_CATEGORIA[categoria]
        for texto in textos:
            for _ in range(repetir):
                registros.append({
                    "id": id_counter,
                    "texto": texto,
                    "categoria": categoria,
                    "sentimento": sentimento,
                    "nota": gerar_nota(sentimento),
                    "canal": gerar_canal(),
                    "data": gerar_data_aleatoria(),
                })
                id_counter += 1

    random.shuffle(registros)

    # Reatribuir IDs após embaralhamento
    for i, r in enumerate(registros, start=1):
        r["id"] = i

    return pd.DataFrame(registros)


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    import os

    output_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "atendimentos.csv")

    df = gerar_base(repetir=4)

    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("BASE DE DADOS GERADA COM SUCESSO")
    print("=" * 60)
    print(f"Arquivo salvo em: {output_path}")
    print(f"Total de registros: {len(df)}")
    print()
    print("Distribuição por categoria:")
    dist = df["categoria"].value_counts()
    for cat, n in dist.items():
        print(f"  {cat:<45} {n:>4} registros")
    print()
    print("Distribuição por sentimento:")
    for sent, n in df["sentimento"].value_counts().items():
        print(f"  {sent:<12} {n:>4} registros")
    print("=" * 60)

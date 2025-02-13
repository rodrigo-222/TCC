Fazer aplicativo android para tratar audio com filtros adaptativos.

Filtros Adaptativos são filtros com coeficientes que variam no tempo, capazes de se adaptar automaticamente a mudanças no seu sinal de entrada.

Objetivo:
    Processamento de audio adaptativo


Aplicativo que roda em segundo plano para coletar dados em tempo real para ruídos de teste.
    -Problemas:
        Armazenamento desses dados e a transferencia desses dados.
        Compartilhamento do microfone enquanto ele roda no background com outros aplicativos.
        Classificação dos tipos de ruídos para cada situação.
            - Tipos de ruídos de áudio e suas definições:

                1. Ruído Branco: Ruido gausseano
                    Definição: Ruído com uma distribuição uniforme de energia em todas as frequências audíveis.
                    Características: Possui uma amplitude constante em todas as frequências, o que resulta em um som contínuo e uniforme.

                1. Ruído Rosa: Ruído de 1/f
                    Definição: Ruído com uma distribuição de energia que diminui proporcionalmente à frequência.
                    Características: Possui uma amplitude que diminui à medida que a frequência aumenta, resultando em um som mais suave e agradável.

                3. Ruído Marrom: Ruído de 1/f^2
                    Definição: ruído que possui mais energia em frequências baixas em comparação ao ruído rosa e ao ruído branco.
                    Características: A potência é mais concentrada em frequências mais baixas, resultando em um som mais suave e profundo.

                4. Ruído de Cliques e Estalos:
                    Definição: Ruído caracterizado por pequenos estalos ou cliques no áudio.
                    Características: Pode ser causado por problemas na gravação, como falhas no equipamento ou interferências externas.

    -Soluções:
        Limitar o tamanho do audio ou criar um interfaçe que determine quando ligar o aplicativo.

Software para fazer o tratamento e apresentar os graficamente as diferenças resultados.
    Algortimos utilizados:
        1. Algoritmo de Mínimos Quadrados Médios (LMS): 
            Este algoritmo ajusta os coeficientes do filtro com base na diferença entre a saída desejada e a saída real. 
            Ele atualiza os coeficientes de uma forma que minimiza o erro médio quadrático.
        2. Algoritmo de Mínimos Quadrados Médios Normalizados (NLMS): 
            Semelhante ao algoritmo LMS, o algoritmo NLMS ajusta os coeficientes do filtro com base no erro entre as saídas desejada e real. 
            No entanto, ele normaliza a etapa de atualização para evitar grandes atualizações nos coeficientes.
        3. Algoritmo de Mínimos Quadrados Recursivos (RLS): 
            O algoritmo RLS estima recursivamente os coeficientes do filtro minimizando a soma dos erros quadráticos. 
            Ele utiliza uma técnica de inversão de matriz para atualizar os coeficientes.

    As melhores situações para cada algoritmo de filtragem adaptativa são:

        1. Algoritmo de Mínimos Quadrados Médios (LMS):

            É adequado para situações em que o sinal de entrada é estacionário e a matriz de correlação é desconhecida.
            É amplamente utilizado em aplicações de cancelamento de eco, equalização de canal e cancelamento de ruído.

        2. Algoritmo de Mínimos Quadrados Médios Normalizados (NLMS):

            É adequado para situações em que a variação da energia do sinal de entrada é alta.
            É útil quando a matriz de correlação é desconhecida ou varia com o tempo.
            Evita grandes atualizações nos coeficientes, o que pode ser benéfico em aplicações em que a convergência rápida é desejada.

        3. Algoritmo de Mínimos Quadrados Recursivos (RLS):

            É adequado para situações em que a matriz de correlação é conhecida ou pode ser estimada com precisão.
            É útil quando a matriz de correlação varia lentamente com o tempo.
            Oferece uma convergência mais rápida e uma melhor capacidade de rastreamento de mudanças no sistema em comparação com o algoritmo LMS.

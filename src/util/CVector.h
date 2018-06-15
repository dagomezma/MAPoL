#ifndef CVECTOR_H_
#define CVECTOR_H_

/*
 * Estrutura que empacota um vetor padrão da linguagem C.
 * Não define funções para tratar o vetor
 */
template<typename T>
struct CVector {

    /*
     * Número de elementos do vetor
     */
    unsigned int len;

    /*
     * Ponteiro para o vetor
     */
    T* ptr;
};

#endif /* CVECTOR_H_ */
